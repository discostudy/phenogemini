#  Copyright (c) 2018-2023 Beijing Ekitech Co., Ltd.
#  All rights reserved.

import glob
import logging
import multiprocessing
import os
import sqlite3
import sys

import zstandard
from lxml import etree

from ek_phenopub.util import get_usable_cpu_count

logger = logging.getLogger(__name__)

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s %(threadName)s %(message)s')


def pack_worker(q_in, q_out, dict_file):
    zstd_level = 19
    with open(dict_file, 'rb') as f:
        zstd_dict_bytes = f.read()

    compression_dict = zstandard.ZstdCompressionDict(zstd_dict_bytes)
    compression_dict.precompute_compress(level=zstd_level)
    cctx = zstandard.ZstdCompressor(level=zstd_level, dict_data=compression_dict, write_dict_id=False)

    # Loop indefinitely, reading from the queue and inserting into the database
    while True:
        # Get the next item from the queue
        item = q_in.get()
        if item is None:
            break

        pmc_id, content = item
        q_out.put((pmc_id, cctx.compress(content)))


def db_writer_process(q, db_file):
    # Open a connection to the SQLite database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("CREATE TABLE pub_content (id INTEGER PRIMARY KEY AUTOINCREMENT, pub_id text NOT NULL, content BLOB NULL);")
    conn.commit()

    # Loop indefinitely, reading from the queue and inserting into the database
    while True:
        # Get the next item from the queue
        item = q.get()
        if item is None:
            break

        pmc_id, content = item
        c.execute("INSERT INTO pub_content (pub_id, content) VALUES (?, ?);", (pmc_id, content))

    c.execute("CREATE UNIQUE INDEX idx_pub_content_pub_id ON pub_content (pub_id);")
    conn.commit()

    # Close the database connection
    conn.close()


def extract_pubmed_id(elem):
    pmid1 = elem.find('MedlineCitation/PMID').text.strip()
    article_ids = [art for art in elem.findall('PubmedData/ArticleIdList/ArticleId') if art.attrib['IdType'] == 'pubmed']
    if len(article_ids) != 1:
        raise RuntimeError(f'Multiple ArticleId for {pmid1}')

    pmid2 = article_ids[0].text.strip()
    if pmid1 != pmid2:
        raise RuntimeError(f'Inconsistent PMID, pmid1={pmid1}, pmid2={pmid2}')

    return pmid1


def iter_pubmed_xml_file(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    articles = root.findall('PubmedArticle')
    for article in articles:
        pmid = 'PM:' + extract_pubmed_id(article)
        yield pmid, etree.tostring(article)


def pub_reader(q, source_path, file_mappings):
    logger.info(f'Processing source file: {source_path}')
    source_filename = os.path.basename(source_path)

    for pmid, content in iter_pubmed_xml_file(source_path):
        mapping_val = file_mappings.get(pmid)
        if mapping_val is None:
            file_mappings[pmid] = source_filename
        else:
            logger.info(f'Already exist, not adding to db: {pmid}, orig: {mapping_val}, new: {source_filename}')
            continue

        q.put((pmid, content))


def pack_pub_to_db(glob_pat, db_file, dict_file, nprocess=0):
    if nprocess == 0:
        nprocess = get_usable_cpu_count()

    db_file = '/project/phenopub/data/pubmed/pubmed-store.db'
    glob_pat = '/home/chenzefu/phenoapt_nlp/pubmed/pubmed*.xml'
    dict_file = '/project/phenopub/data/pubmed/pubmed_xml_dictionary.4m'

    # Create the queue and start the consumer process
    q_in = multiprocessing.Queue(10_000)
    q_db = multiprocessing.Queue(10_000)

    consumers = []
    for idx in range(nprocess):
        consumer = multiprocessing.Process(target=pack_worker, args=(q_in, q_db, dict_file))
        logger.info(f'Starting producer {idx}')
        consumer.start()
        consumers.append(consumer)

    logger.info('Starting db writer')
    db_writer = multiprocessing.Process(target=db_writer_process, args=(q_db, db_file))
    db_writer.start()

    logger.info('Starting producer')
    file_mappings = {}

    for xml_file in glob.glob(glob_pat):
        pub_reader(q_in, xml_file, file_mappings)
        break  # TODO

    for _ in range(nprocess):
        q_in.put(None)

    logger.info('Waiting for consumers to terminate')
    for consumer in consumers:
        consumer.join()
    logger.info('Consumers terminated')

    logger.info('Waiting for db writer to terminate')
    q_db.put(None)
    db_writer.join()
    logger.info('db writer terminated')

    logger.info('Writing dictionary file')
    with open(dict_file, 'rb') as f:
        zstd_dict_bytes = f.read()

    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("CREATE TABLE pub_dictionary (dict_id integer PRIMARY KEY, dict_content blob NOT NULL);")
    conn.commit()
    c.execute("INSERT INTO pub_dictionary (dict_id, dict_content) VALUES (?, ?);", (0, zstd_dict_bytes))
    conn.commit()
    conn.close()
    logger.info('pub-store... DONE')


class PubRepository:
    def __init__(self, pub_db):
        pub_db_path = os.path.abspath(pub_db)
        self.conn = sqlite3.connect(f'file:{pub_db_path}?mode=ro', uri=True)
        self.cur = self.conn.cursor()
        self._dctx = None

    @property
    def dctx(self):
        if self._dctx is None:
            zstd_dict_bytes = self.cur.execute("SELECT dict_content FROM pub_dictionary WHERE dict_id = ?;", (0,)).fetchone()[0]
            compression_dict = zstandard.ZstdCompressionDict(zstd_dict_bytes)
            self._dctx = zstandard.ZstdDecompressor(dict_data=compression_dict)

        return self._dctx

    def get_pub_info(self, pub_id):
        info = self.cur.execute("SELECT id, pub_id FROM pub_content WHERE pub_id = ?;", (pub_id,)).fetchone()
        if info is None:
            return None
        return dict(zip(['id', 'pub_id'], info))

    def get_pub_info_by_id(self, id_):
        info = self.cur.execute("SELECT id, pub_id FROM pub_content WHERE id = ?;", (id_,)).fetchone()
        if info is None:
            return None
        return dict(zip(['id', 'pub_id'], info))

    def get_pub_xml(self, pub_id):
        blob = self.get_pub_bytes(pub_id)
        if blob is None:
            return None
        return self.decode_text(blob)

    def get_pub_bytes(self, pub_id):
        blob = self.cur.execute("SELECT content FROM pub_content WHERE pub_id = ?;", (pub_id,)).fetchone()
        if blob is None:
            return None
        return self.dctx.decompress(blob[0])

    def get_pub_full(self, pub_id):
        sql = "SELECT a.id, a.pub_id, a.content FROM pub_content a WHERE a.pub_id = ?;"
        rec = self.cur.execute(sql, (pub_id,)).fetchone()
        if rec is None:
            return None

        id_, pub_id, blob = rec
        text = self.decode_text(self.dctx.decompress(blob))
        return dict(id=id_, pub_id=pub_id, text=text)
    
    def get_pub_full_by_id(self, id_):
        sql = "SELECT a.id, a.pub_id, a.content FROM pub_content a WHERE a.id = ?;"
        rec = self.cur.execute(sql, (id_,)).fetchone()
        if rec is None:
            return None

        id_, pub_id, blob = rec
        text = self.decode_text(self.dctx.decompress(blob))
        return dict(id=id_, pub_id=pub_id, text=text)
    
    def extract_jtak(self, input_text):
        input_xml = etree.fromstring(input_text).find("MedlineCitation/Article")
        journal_content = input_xml.find("Journal")[2].text
        title_content = input_xml.find("ArticleTitle").text
        keywords = input_xml.xpath('//Keyword')
        keyword_content = []
        for keyword in keywords:
            if keyword.text is not None:  # 确保keyword.text不是None
                keyword_content.append(keyword.text)
        keyword_content = ", ".join(keyword_content)
        abstract = input_xml.xpath("//AbstractText")
        if abstract != None:
            abstract_content = [(x.attrib["Label"] + "\n" + (x.text or "")) if x.attrib else ((x.text or "") + "\n") for x in abstract]
            return "Journal:" + "\n" + (journal_content or "") + "\n\n" + "Title:" + "\n" + (title_content or "") + "\n\n" + "Abstract:" + "\n" + "\n".join(abstract_content) + "\n\n" + "Keywords:" + "\n" + (keyword_content or "")
        if abstract == None:
            return "Journal:" + "\n" + (journal_content or "") + "\n\n" + "Title:" + "\n" + (title_content or "") + "\n\n" + "Abstract:" + "\n" + "Keywords:" + "\n" + (keyword_content or "")
        
    def extract_jta(self, input_text):
        input_xml = etree.fromstring(input_text).find("MedlineCitation/Article")
        journal_content = input_xml.find("Journal")[2].text
        title_content = input_xml.find("ArticleTitle").text
        abstract = input_xml.xpath("//AbstractText")
        if abstract != None:
            abstract_content = [(x.attrib["Label"] + "\n" + (x.text or "")) if x.attrib else ((x.text or "") + "\n") for x in abstract]
            return "Journal:" + "\n" + (journal_content or "") + "\n\n" + "Title:" + "\n" + (title_content or "") + "\n\n" + "Abstract:" + "\n" + "\n".join(abstract_content)
        if abstract == None:
            return "Journal:" + "\n" + (journal_content or "") + "\n\n" + "Title:" + "\n" + (title_content or "") + "\n\n" + "Abstract:"

    def get_pub_jtak_by_id(self, id_):
        sql = "SELECT a.id, a.pub_id, a.content FROM pub_content a WHERE a.id = ?;"
        rec = self.cur.execute(sql, (id_,)).fetchone()
        if rec is None:
            return None

        id_, pub_id, blob = rec
        text = self.decode_text(self.dctx.decompress(blob))
        jtak = self.extract_jtak(text)
        return dict(id=id_, pub_id=pub_id, text=jtak)
    
    def get_pub_jta_by_id(self, id_):
        sql = "SELECT a.id, a.pub_id, a.content FROM pub_content a WHERE a.id = ?;"
        rec = self.cur.execute(sql, (id_,)).fetchone()
        if rec is None:
            return None

        id_, pub_id, blob = rec
        text = self.decode_text(self.dctx.decompress(blob))
        jta = self.extract_jta(text)
        return dict(id=id_, pub_id=pub_id, text=jta)
    
    def get_all_pub_ids(self):
        all_info = self.cur.execute("SELECT pub_id FROM pub_content ORDER BY pub_id;").fetchall()
        return [x[0] for x in all_info]
    

    @staticmethod
    def decode_text(text_bytes):
        return text_bytes.decode('utf-8')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.conn.close()


if __name__ == '__main__':
    pack_pub_to_db(None, None, None, nprocess=50)
