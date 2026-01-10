"""
Department-based FAISS index routing service.
Routes retrieval to department-specific FAISS indices based on request payload or inferred department.
"""
import os
import logging
from typing import Optional, Dict, List
from langchain_community.vectorstores import FAISS
from services.llm import get_llm
from langchain.prompts import PromptTemplate
from services.multimodal_embeddings import get_multimodal_embeddings

logger = logging.getLogger(__name__)

VALID_DEPARTMENTS = ["structures", "avionics", "propulsion", "maintenance", "general", "default"]
BASE_INDEX_PATH = "snag_faiss_index"


def infer_department(query: str) -> str:
    """
    Infer department from query using lightweight LLM classifier.
    Used ONLY for routing - inference is not persisted.
    
    Args:
        query: User query string
    
    Returns:
        One of: structures, avionics, propulsion, maintenance, general
    """
    try:
        llm = get_llm()
        
        prompt = PromptTemplate.from_template("""
Classify the following aircraft maintenance query into exactly one department.

DEPARTMENTS:
- structures: Structural components, airframe, fuselage, wings, landing gear
- avionics: Electronic systems, navigation, communication, radar, displays
- propulsion: Engines, fuel systems, power plants, turbines
- maintenance: General maintenance procedures, inspections, tools
- general: General queries that don't fit specific departments

QUERY: {query}

Respond with ONLY one word: structures, avionics, propulsion, maintenance, or general.
Do not include explanations, just the department name.
""")
        
        response = llm.invoke(prompt.format(query=query))
        inferred_dept = response.strip().lower() if isinstance(response, str) else str(response).strip().lower()
        
        # Validate inferred department
        if inferred_dept not in VALID_DEPARTMENTS:
            logger.warning(f"Invalid inferred department: {inferred_dept}, falling back to general")
            inferred_dept = "general"
        
        logger.info(f"Inferred department: {inferred_dept} for query: {query[:50]}...")
        return inferred_dept
        
    except Exception as e:
        logger.error(f"Error inferring department: {str(e)}, falling back to general")
        return "general"


def get_department_index_path(department: str) -> str:
    """
    Get FAISS index path for a department.
    
    Args:
        department: Department name
    
    Returns:
        Path to department FAISS index directory
    """
    if department == "default":
        return BASE_INDEX_PATH
    
    dept_normalized = department.lower().strip()
    if dept_normalized not in VALID_DEPARTMENTS:
        logger.warning(f"Invalid department: {department}, falling back to general")
        dept_normalized = "general"
    
    return os.path.join(BASE_INDEX_PATH, dept_normalized, "faiss_index")


def load_department_faiss(department: str) -> Optional[FAISS]:
    """
    Load FAISS index for a specific department.
    Falls back to general if department index doesn't exist.
    
    Args:
        department: Department name
    
    Returns:
        FAISS index or None if no index exists
    """
    index_path = get_department_index_path(department)
    
    # Check if index exists
    if not os.path.exists(index_path):
        logger.warning(f"Department index not found: {index_path}")
        if department != "general":
            logger.info(f"Falling back to general index")
            return load_department_faiss("general")
        else:
            logger.error(f"General index not found: {index_path}")
            return None
    
    try:
        embeddings = get_multimodal_embeddings(model_path="./all-MiniLM-L6-v2", device='cpu')
        db = FAISS.load_local(
            index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"✓ Loaded FAISS index: {index_path}")
        return db
    except Exception as e:
        logger.error(f"Error loading department FAISS index {index_path}: {str(e)}")
        if department != "general":
            logger.info(f"Falling back to general index")
            return load_department_faiss("general")
        return None


def load_all_departments_faiss() -> Optional[FAISS]:
    """
    Load and merge FAISS indices from all departments.
    Used when department="default".
    
    Returns:
        Merged FAISS index or None if no indices exist
    """
    all_dbs = []
    loaded_departments = []
    
    for dept in ["structures", "avionics", "propulsion", "maintenance", "general"]:
        index_path = get_department_index_path(dept)
        if os.path.exists(index_path):
            try:
                embeddings = get_multimodal_embeddings(model_path="./all-MiniLM-L6-v2", device='cpu')
                db = FAISS.load_local(
                    index_path,
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
                all_dbs.append(db)
                loaded_departments.append(dept)
                logger.info(f"✓ Loaded department index: {dept}")
            except Exception as e:
                logger.warning(f"Failed to load department index {dept}: {str(e)}")
    
    if not all_dbs:
        logger.error("No department indices found")
        return None
    
    if len(all_dbs) == 1:
        logger.info(f"Using single department index: {loaded_departments[0]}")
        return all_dbs[0]
    
    # Merge all indices
    try:
        merged_db = all_dbs[0]
        for db in all_dbs[1:]:
            merged_db.merge_from(db)
        logger.info(f"✓ Merged FAISS indices from {len(loaded_departments)} departments: {', '.join(loaded_departments)}")
        return merged_db
    except Exception as e:
        logger.error(f"Error merging department indices: {str(e)}")
        return all_dbs[0] if all_dbs else None


def route_to_department_faiss(department: Optional[str], query: Optional[str] = None) -> tuple:
    """
    Route to appropriate FAISS index based on department or inferred department.
    
    Args:
        department: Department name (can be None)
        query: User query (used for inference if department is None)
    
    Returns:
        Tuple of (FAISS index, actual_department_used)
    """
    actual_department = department
    
    # Determine department
    if actual_department is None or actual_department == "":
        # When no department is selected, always use "general" index (no inference)
        actual_department = "general"
        logger.info(f"📂 Routing: department=None/null → using general index")
    else:
        actual_department = actual_department.lower().strip()
        logger.info(f"📂 Routing: department={actual_department} (from request)")
    
    # Handle default case - merge all departments
    if actual_department == "default":
        logger.info(f"📂 Routing: default → loading ALL department indices")
        db = load_all_departments_faiss()
        return db, "default"
    
    # Load specific department index
    db = load_department_faiss(actual_department)
    
    if db is None:
        logger.error(f"Failed to load any FAISS index for department: {actual_department}")
        return None, actual_department
    
    logger.info(f"✓ Routed to department: {actual_department}")
    return db, actual_department

