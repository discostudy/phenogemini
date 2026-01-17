"""
数据验证模块
"""
from typing import Dict, Any, List, Tuple

def validate_extraction_result(data: Dict[Any, Any]) -> Tuple[bool, List[str]]:
    """
    验证提取结果的JSON结构是否符合预期
    
    Returns:
        Tuple[bool, List[str]]: (是否有效, 错误列表)
    """
    errors = []
    
    if not isinstance(data, dict):
        errors.append("顶层数据必须是字典格式")
        return False, errors
    
    # 检查顶层结构
    required_top_keys = ["document_info", "patients"]
    for key in required_top_keys:
        if key not in data:
            errors.append(f"缺少必需字段: {key}")
    
    # 检查document_info结构
    if "document_info" in data:
        doc_info = data["document_info"]
        if not isinstance(doc_info, dict):
            errors.append("document_info必须是字典格式")
        else:
            required_doc_keys = ["pmid", "total_patients_found"]
            for key in required_doc_keys:
                if key not in doc_info:
                    errors.append(f"document_info缺少必需字段: {key}")
            
            # 检查数据类型
            if "total_patients_found" in doc_info:
                if not isinstance(doc_info["total_patients_found"], (int, float)):
                    errors.append("total_patients_found必须是数字")
    
    # 检查患者数据结构
    if "patients" in data:
        if not isinstance(data["patients"], list):
            errors.append("patients字段必须是列表")
        else:
            for i, patient in enumerate(data["patients"]):
                patient_errors = validate_patient_data(patient, i)
                errors.extend(patient_errors)
    
    return len(errors) == 0, errors

def validate_patient_data(patient: Any, index: int) -> List[str]:
    """验证单个患者数据"""
    errors = []
    
    if not isinstance(patient, dict):
        errors.append(f"患者{index}数据必须是字典格式")
        return errors
    
    # 检查必需字段
    required_patient_keys = ["patient_id", "phenotype", "genotype", "demographics"]
    for key in required_patient_keys:
        if key not in patient:
            errors.append(f"患者{index}缺少必需字段: {key}")
    
    # 检查phenotype结构
    if "phenotype" in patient:
        phenotype = patient["phenotype"]
        if isinstance(phenotype, dict):
            if "description" not in phenotype:
                errors.append(f"患者{index}的phenotype缺少description字段")
        else:
            errors.append(f"患者{index}的phenotype必须是字典格式")
    
    # 检查genotype结构
    if "genotype" in patient:
        genotype = patient["genotype"]
        if isinstance(genotype, dict):
            if "variants" in genotype:
                if not isinstance(genotype["variants"], list):
                    errors.append(f"患者{index}的variants必须是列表")
                else:
                    for j, variant in enumerate(genotype["variants"]):
                        variant_errors = validate_variant_data(variant, index, j)
                        errors.extend(variant_errors)
        else:
            errors.append(f"患者{index}的genotype必须是字典格式")
    
    # 检查demographics结构
    if "demographics" in patient:
        demographics = patient["demographics"]
        if not isinstance(demographics, dict):
            errors.append(f"患者{index}的demographics必须是字典格式")
    
    return errors

def validate_variant_data(variant: Any, patient_index: int, variant_index: int) -> List[str]:
    """验证变异数据"""
    errors = []
    
    if not isinstance(variant, dict):
        errors.append(f"患者{patient_index}的变异{variant_index}必须是字典格式")
        return errors
    
    # 检查基本字段
    basic_fields = ["gene", "zygosity", "testing_method"]
    for field in basic_fields:
        if field not in variant:
            errors.append(f"患者{patient_index}的变异{variant_index}缺少字段: {field}")
    
    # 检查HGVS命名法格式（基本检查）
    hgvs_fields = ["hgvs_cdna", "hgvs_protein", "hgvs_genomic"]
    for field in hgvs_fields:
        if field in variant and variant[field] != "not_reported":
            value = variant[field]
            if isinstance(value, str) and value:
                # 基本HGVS格式检查
                if field == "hgvs_cdna" and not (value.startswith("c.") or value == "not_reported"):
                    errors.append(f"患者{patient_index}的变异{variant_index} {field} 格式可能不正确: {value}")
                elif field == "hgvs_protein" and not (value.startswith("p.") or value == "not_reported"):
                    errors.append(f"患者{patient_index}的变异{variant_index} {field} 格式可能不正确: {value}")
                elif field == "hgvs_genomic" and not (value.startswith("g.") or value == "not_reported"):
                    errors.append(f"患者{patient_index}的变异{variant_index} {field} 格式可能不正确: {value}")
    
    # 检查基因组坐标结构
    if "genomic_coordinate" in variant:
        coord = variant["genomic_coordinate"]
        if isinstance(coord, dict):
            coord_fields = ["chromosome", "position", "reference_genome"]
            for field in coord_fields:
                if field not in coord:
                    errors.append(f"患者{patient_index}的变异{variant_index}的genomic_coordinate缺少字段: {field}")
        else:
            errors.append(f"患者{patient_index}的变异{variant_index}的genomic_coordinate必须是字典格式")
    
    # 检查inheritance结构
    if "inheritance" in variant:
        inheritance = variant["inheritance"]
        if isinstance(inheritance, dict):
            inheritance_fields = ["de_novo", "maternal", "paternal"]
            for field in inheritance_fields:
                if field not in inheritance:
                    errors.append(f"患者{patient_index}的变异{variant_index}的inheritance缺少字段: {field}")
        else:
            errors.append(f"患者{patient_index}的变异{variant_index}的inheritance必须是字典格式")
    
    return errors

def validate_hgvs_nomenclature(hgvs_string: str, variant_type: str) -> bool:
    """
    验证HGVS命名法格式
    
    Args:
        hgvs_string: HGVS字符串
        variant_type: 变异类型 ('cdna', 'protein', 'genomic')
    
    Returns:
        bool: 是否符合格式
    """
    if hgvs_string == "not_reported" or not hgvs_string:
        return True
    
    if variant_type == "cdna":
        return hgvs_string.startswith("c.")
    elif variant_type == "protein":
        return hgvs_string.startswith("p.")
    elif variant_type == "genomic":
        return hgvs_string.startswith("g.")
    
    return False

def validate_zygosity(zygosity: str) -> bool:
    """验证合子状态"""
    valid_zygosity = [
        "homozygous", 
        "heterozygous", 
        "compound_heterozygous", 
        "not_reported"
    ]
    return zygosity in valid_zygosity

def validate_testing_method(method: str) -> bool:
    """验证检测方法"""
    valid_methods = [
        "WES", "WGS", "Panel", "Sanger", 
        "not_reported", "unknown"
    ]
    return method in valid_methods

def get_quality_score(data: Dict[Any, Any]) -> float:
    """
    计算提取质量评分 (0-1)
    """
    if not data.get("patients"):
        return 0.0
    
    total_score = 0.0
    total_patients = len(data["patients"])
    
    for patient in data["patients"]:
        patient_score = 0.0
        max_score = 0.0
        
        # 表型信息完整性 (30%)
        if patient.get("phenotype", {}).get("description", "").strip():
            if patient["phenotype"]["description"] != "not_reported":
                patient_score += 0.3
        max_score += 0.3
        
        # 基因型信息完整性 (50%)
        variants = patient.get("genotype", {}).get("variants", [])
        if variants:
            variant_scores = []
            for variant in variants:
                variant_score = 0.0
                variant_max = 0.0
                
                # 基因名称
                if variant.get("gene", "").strip() and variant["gene"] != "not_reported":
                    variant_score += 0.1
                variant_max += 0.1
                
                # HGVS命名
                hgvs_fields = ["hgvs_cdna", "hgvs_protein"]
                for field in hgvs_fields:
                    if variant.get(field, "").strip() and variant[field] != "not_reported":
                        variant_score += 0.15
                    variant_max += 0.15
                
                # 合子状态
                if validate_zygosity(variant.get("zygosity", "")):
                    variant_score += 0.1
                variant_max += 0.1
                
                if variant_max > 0:
                    variant_scores.append(variant_score / variant_max)
            
            if variant_scores:
                patient_score += 0.5 * (sum(variant_scores) / len(variant_scores))
        max_score += 0.5
        
        # 人口学信息完整性 (20%)
        demographics = patient.get("demographics", {})
        demo_fields = ["ethnicity", "nationality"]
        demo_score = 0
        for field in demo_fields:
            if demographics.get(field, "").strip() and demographics[field] != "not_reported":
                demo_score += 1
        if demo_score > 0:
            patient_score += 0.2 * (demo_score / len(demo_fields))
        max_score += 0.2
        
        if max_score > 0:
            total_score += patient_score / max_score
    
    return total_score / total_patients if total_patients > 0 else 0.0