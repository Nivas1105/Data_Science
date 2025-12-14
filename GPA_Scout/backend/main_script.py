"""
UTA Course Q&A Agent - Production Grade with Enhanced Analytics
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


import re
import logging
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# ========================
# 2. CONFIGURATION MANAGEMENT
# ========================
@dataclass
class ModelConfig:
    """Configuration for all models used in the system."""
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    embed_model_id: str = "BAAI/bge-base-en-v1.5"
    torch_dtype: torch.dtype = torch.float32  
    device_map: str = "cpu"  
    max_new_tokens: int = 50 
    temperature: float = 0.1
    top_k: int = 3

@dataclass
class DataConfig:
    data_file: str = "/Users/nivasm/Documents/deploy_ml/project_data.csv"
    index_prefix: str = "uta_production"
    chunk_sizes: Dict[str, int] = field(default_factory=lambda: {
        'courses': 3,
        'professors': 3,
        'sections': 3
    })

@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    log_level: str = "INFO"
    cache_size: int = 1000

# ========================
# 3. LOGGING SETUP
# ========================
class ProductionLogger:

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger

# ========================
# 4. ERROR HANDLING
# ========================
class CourseQAError(Exception):
    pass

class DataLoadingError(CourseQAError):
    pass

class ModelLoadingError(CourseQAError):
    pass

class IndexBuildingError(CourseQAError):
    pass

# ========================
# 5. DATA PROCESSOR (ENHANCED WITH GRADE ANALYTICS)
# ========================
class DataProcessor:
    def __init__(self, config: DataConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.raw_df = None
        self.df_courses = None
        self.df_professors = None
        self.course_title_map = {}  # Cache for course titles

    def load_and_validate_data(self) -> None:
        try:
            self.logger.info(f"Loading data from {self.config.data_file}")

            if not os.path.exists(self.config.data_file):
                raise DataLoadingError(f"Data file not found: {self.config.data_file}")

            self.raw_df = pd.read_csv(self.config.data_file)
            self._validate_dataframe()
            self._preprocess_data()
            self._create_aggregations()
            self._build_course_title_map()
            self._enhance_professor_analytics()

            self.logger.info("Data loading and validation completed successfully")

        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise DataLoadingError(f"Failed to load data: {str(e)}")

    def _validate_dataframe(self) -> None:
        required_columns = ['Subject', 'Catalog Number', 'Course Name', 'Primary Instructor First Name',
                           'Primary Instructor Last Name', 'Description', 'Term', 'A', 'B', 'C', 'D', 'F', 'Total Grades']

        missing_columns = [col for col in required_columns if col not in self.raw_df.columns]
        if missing_columns:
            raise DataLoadingError(f"Missing required columns: {missing_columns}")

        optional_columns = ['Course Career', 'GRAD']
        for col in optional_columns:
            if col not in self.raw_df.columns:
                self.logger.warning(f"Optional column '{col}' not found in dataset")

        if len(self.raw_df) == 0:
            raise DataLoadingError("Dataframe is empty")

    def _preprocess_data(self) -> None:
        numeric_cols = ['A', 'B', 'C', 'D', 'F', 'W', 'P', 'I', 'Q', 'Z', 'R', 'Total Grades']
        for col in numeric_cols:
            if col in self.raw_df.columns:
                self.raw_df[col] = pd.to_numeric(self.raw_df[col], errors='coerce').fillna(0)

        text_cols = ['Primary Instructor First Name', 'Primary Instructor Last Name',
                    'Description', 'Course Name', 'Term', 'Course Career', 'GRAD']
        for col in text_cols:
            if col in self.raw_df.columns:
                self.raw_df[col] = self.raw_df[col].fillna("").astype(str)
        self.raw_df['course_code'] = (
            self.raw_df['Subject'].astype(str).str.strip() + ' ' +
            self.raw_df['Catalog Number'].astype(str).str.strip()
        )
        self.raw_df['instructor'] = (
            self.raw_df['Primary Instructor First Name'] + ' ' +
            self.raw_df['Primary Instructor Last Name']
        ).str.strip()

        self.raw_df['graded_total'] = self.raw_df[['A', 'B', 'C', 'D', 'F']].sum(axis=1)
        gpa_numerator = (self.raw_df['A'] * 4 + self.raw_df['B'] * 3 +
                        self.raw_df['C'] * 2 + self.raw_df['D'] * 1)
        self.raw_df['gpa'] = (gpa_numerator / self.raw_df['graded_total'].replace(0, np.nan)).fillna(0)
        self.raw_df['a_rate'] = (self.raw_df['A'] / self.raw_df['graded_total'].replace(0, np.nan) * 100).fillna(0)
        self.raw_df['pass_rate'] = ((self.raw_df['A'] + self.raw_df['B'] + self.raw_df['C']) /
                                   self.raw_df['Total Grades'].replace(0, np.nan) * 100).fillna(0)
        self.raw_df['dfw_rate'] = ((self.raw_df['D'] + self.raw_df['F'] + self.raw_df['W']) /
                                  self.raw_df['Total Grades'].replace(0, np.nan) * 100).fillna(0)

    def _create_aggregations(self) -> None:
        """Create aggregated views of the data with enhanced course information."""
        aggregation_dict = {
            'title': ('Course Name', 'first'),
            'description': ('Description', 'first'),
            'avg_gpa': ('gpa', 'mean'),
            'avg_a_rate': ('a_rate', 'mean'),
            'avg_pass_rate': ('pass_rate', 'mean'),
            'avg_dfw_rate': ('dfw_rate', 'mean'),
            'total_students': ('Total Grades', 'sum'),
            'times_offered': ('Term', 'count')
        }

        if 'Course Career' in self.raw_df.columns:
            aggregation_dict['course_career'] = ('Course Career', 'first')
        if 'GRAD' in self.raw_df.columns:
            aggregation_dict['grad'] = ('GRAD', 'first')

        self.df_courses = self.raw_df.groupby('course_code').agg(**aggregation_dict).reset_index()

        professor_df = self.raw_df[self.raw_df['instructor'] != ""].copy()
        self.df_professors = professor_df.groupby('instructor').agg(
            avg_gpa_given=('gpa', 'mean'),
            avg_a_rate=('a_rate', 'mean'),
            avg_pass_rate=('pass_rate', 'mean'),
            avg_dfw_rate=('dfw_rate', 'mean'),
            total_students=('Total Grades', 'sum'),
            courses_taught=('course_code', lambda x: sorted(x.unique().tolist())),
            terms_taught=('Term', 'count')
        ).reset_index()

    def _enhance_professor_analytics(self) -> None:
        for idx, prof_row in self.df_professors.iterrows():
            prof_name = prof_row['instructor']
            prof_courses = self.raw_df[self.raw_df['instructor'] == prof_name]

            if len(prof_courses) > 0:
                # Calculate additional metrics
                total_graded = prof_courses['graded_total'].sum()
                a_plus_b_rate = ((prof_courses['A'].sum() + prof_courses['B'].sum()) /
                               total_graded * 100) if total_graded > 0 else 0

                # Determine teaching style
                if prof_row['avg_gpa_given'] >= 3.6 and prof_row['avg_a_rate'] >= 60:
                    teaching_style = "Generous Grader"
                elif prof_row['avg_gpa_given'] <= 2.8 or prof_row['avg_dfw_rate'] >= 20:
                    teaching_style = "Tough Grader"
                else:
                    teaching_style = "Balanced Grader"

                # Add to dataframe
                self.df_professors.at[idx, 'a_plus_b_rate'] = a_plus_b_rate
                self.df_professors.at[idx, 'teaching_style'] = teaching_style

    def _build_course_title_map(self) -> None:
        """Build a cache of course codes to titles for quick lookup."""
        self.course_title_map = dict(zip(self.df_courses['course_code'], self.df_courses['title']))
        self.logger.info(f"Built course title map with {len(self.course_title_map)} entries")

    def get_course_title(self, course_code: str) -> str:
        """Get course title with validation."""
        course_code = course_code.upper().strip()
        if course_code in self.course_title_map:
            return self.course_title_map[course_code]

        # Try to find close matches
        for code in self.course_title_map:
            if code.replace(" ", "") == course_code.replace(" ", ""):
                return self.course_title_map[code]

        self.logger.warning(f"Course title not found for: {course_code}")
        return "Title not available"

    def validate_course_code(self, course_code: str) -> bool:
        """Validate if course code exists in dataset."""
        course_code = course_code.upper().strip()
        return course_code in self.course_title_map

    def get_specific_grades(self, course_code: str, term: str = None, professor: str = None) -> Dict[str, Any]:
        """Get specific grade counts for a course/term/professor combination."""
        filters = [self.raw_df['course_code'] == course_code.upper()]

        if term:
            # Handle various term formats
            term_lower = term.lower()
            if 'spring' in term_lower:
                season = 'Spring'
            elif 'fall' in term_lower:
                season = 'Fall'
            elif 'summer' in term_lower:
                season = 'Summer'
            else:
                season = None

            year_match = re.search(r'(\d{4})', term)
            year = year_match.group(1) if year_match else None

            if season and year:
                term_filter = (self.raw_df['Term'].str.contains(season, case=False, na=False) &
                             self.raw_df['Term'].str.contains(year, na=False))
                filters.append(term_filter)

        if professor:
            prof_match = self.find_best_professor_match(professor)
            if prof_match:
                filters.append(self.raw_df['instructor'] == prof_match)

        # Apply filters
        mask = filters[0]
        for f in filters[1:]:
            mask = mask & f

        sections = self.raw_df[mask]

        if sections.empty:
            return {"error": "No matching sections found"}

        # Aggregate grades
        result = {
            'course_code': course_code,
            'term': term,
            'professor': professor,
            'total_sections': len(sections),
            'total_students': int(sections['Total Grades'].sum()),
            'grades': {
                'A': int(sections['A'].sum()),
                'B': int(sections['B'].sum()),
                'C': int(sections['C'].sum()),
                'D': int(sections['D'].sum()),
                'F': int(sections['F'].sum()),
                'W': int(sections['W'].sum())
            },
            'avg_gpa': float(sections['gpa'].mean()),
            'instructors': sections['instructor'].unique().tolist(),
            'terms': sections['Term'].unique().tolist()
        }

        # Calculate percentages
        total_graded = sum(result['grades'].values()) - result['grades']['W']
        if total_graded > 0:
            for grade in ['A', 'B', 'C', 'D', 'F']:
                result['grades'][f'{grade}_pct'] = (result['grades'][grade] / total_graded * 100)

        return result

    def find_best_professor_match(self, professor_query: str) -> Optional[str]:
        """Find best professor name match."""
        professors = self.df_professors['instructor'].tolist()
        query_parts = professor_query.lower().split()

        for professor in professors:
            professor_lower = professor.lower()
            if all(part in professor_lower for part in query_parts):
                return professor
        return None

    def get_courses_by_topic(self, topic: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Find courses related to a specific topic."""
        topic_lower = topic.lower()
        matching_courses = []

        for _, course in self.df_courses.iterrows():
            course_text = f"{course['title']} {course['description']}".lower()

            # Check for topic matches
            if (topic_lower in course_text or
                any(word in course_text for word in topic_lower.split())):

                matching_courses.append({
                    'course_code': course['course_code'],
                    'title': course['title'],
                    'avg_gpa': course['avg_gpa'],
                    'avg_pass_rate': course['avg_pass_rate'],
                    'total_students': course['total_students'],
                    'description': course['description'][:200] + "..." if len(course['description']) > 200 else course['description']
                })

        # Sort by GPA (highest first) for "easiest" queries
        if 'easy' in topic_lower or 'easiest' in topic_lower:
            matching_courses.sort(key=lambda x: x['avg_gpa'], reverse=True)
        elif 'hard' in topic_lower or 'hardest' in topic_lower:
            matching_courses.sort(key=lambda x: x['avg_gpa'])
        else:
            # Default sort by relevance (GPA as proxy for popularity/quality)
            matching_courses.sort(key=lambda x: x['avg_gpa'], reverse=True)

        return matching_courses[:max_results]

    def compare_courses(self, course_codes: List[str]) -> Dict[str, Any]:
        """Compare multiple courses side by side."""
        comparison_data = {}

        for course_code in course_codes:
            course_data = self.df_courses[self.df_courses['course_code'] == course_code]
            if not course_data.empty:
                course = course_data.iloc[0]
                comparison_data[course_code] = {
                    'title': course['title'],
                    'avg_gpa': course['avg_gpa'],
                    'avg_pass_rate': course['avg_pass_rate'],
                    'avg_dfw_rate': course['avg_dfw_rate'],
                    'total_students': course['total_students'],
                    'times_offered': course['times_offered'],
                    'description': course['description']
                }

        return comparison_data

    def create_search_chunks(self) -> Dict[str, pd.DataFrame]:
        """Create optimized search chunks for retrieval with enhanced information."""
        # Course chunks with career and grad information
        self.df_courses['search_chunk'] = self.df_courses.apply(
            lambda row: self._create_course_search_chunk(row), axis=1
        )

        self.df_courses['display_chunk'] = self.df_courses.apply(
            lambda row: self._create_course_display_chunk(row), axis=1
        )

        # Enhanced professor chunks with analytics
        self.df_professors['search_chunk'] = self.df_professors.apply(
            lambda row: self._create_professor_search_chunk(row), axis=1
        )

        self.df_professors['display_chunk'] = self.df_professors.apply(
            lambda row: self._create_professor_display_chunk(row), axis=1
        )

        # Section chunks with enhanced information
        self.raw_df['search_chunk'] = self.raw_df.apply(
            lambda row: self._create_section_search_chunk(row), axis=1
        )

        self.raw_df['display_chunk'] = self.raw_df.apply(
            lambda row: self._create_section_display_chunk(row), axis=1
        )

        return {
            'courses': self.df_courses,
            'professors': self.df_professors,
            'sections': self.raw_df
        }

    def _create_course_search_chunk(self, row: pd.Series) -> str:
        """Create search chunk for course information with career and grad data."""
        base_text = (
            f"Course: {row['course_code']} {row['title']}. "
            f"Description: {row['description']}. "
            f"Average GPA: {row['avg_gpa']:.2f}. "
            f"Pass Rate: {row['avg_pass_rate']:.1f}%. "
            f"Total Students: {row['total_students']}."
        )

        # Add career information if available
        if 'course_career' in row and row['course_career'] and row['course_career'] != "nan":
            base_text += f" Course Career: {row['course_career']}."

        # Add GRAD information if available
        if 'grad' in row and row['grad'] and row['grad'] != "nan":
            base_text += f" GRAD: {row['grad']}."

        return base_text

    def _create_course_display_chunk(self, row: pd.Series) -> str:
        """Create detailed display chunk for course information with all metadata."""
        # Base course information
        display_parts = [
            f"📚 Course: {row['course_code']} - {row['title']}",
            f"📖 Description: {row['description']}",
            f"📊 Overall Stats: Average GPA: {row['avg_gpa']:.2f}, "
            f"Pass Rate: {row['avg_pass_rate']:.1f}%, "
            f"Total Students: {row['total_students']}, "
            f"Times Offered: {row['times_offered']}"
        ]

        # Add career information if available
        if 'course_career' in row and row['course_career'] and row['course_career'] != "nan":
            display_parts.append(f"🎯 Course Career: {row['course_career']}")

        # Add GRAD information if available
        if 'grad' in row and row['grad'] and row['grad'] != "nan":
            display_parts.append(f"🎓 GRAD: {row['grad']}")

        # Add recent history
        history_df = self.raw_df[self.raw_df['course_code'] == row['course_code']].copy()
        if not history_df.empty:
            try:
                history_df['term_year'] = history_df['Term'].str.extract(r'(\d{4})').fillna('0').astype(int)
                recent_terms = history_df.sort_values('term_year', ascending=False).head(5)
            except:
                recent_terms = history_df.tail(5)

            history_lines = []
            for _, term_row in recent_terms.iterrows():
                history_lines.append(
                    f"  - {term_row['Term']}: {term_row['instructor']} (GPA: {term_row['gpa']:.2f}, Students: {term_row['Total Grades']})"
                )
            display_parts.append(f"\n📈 Recent Offerings:\n" + "\n".join(history_lines))

        return "\n".join(display_parts)

    def _create_professor_search_chunk(self, row: pd.Series) -> str:
        """Create search chunk for professor information with analytics."""
        return (
            f"Professor: {row['instructor']}. "
            f"Average GPA: {row['avg_gpa_given']:.2f}. "
            f"A Rate: {row['avg_a_rate']:.1f}%. "
            f"Pass Rate: {row['avg_pass_rate']:.1f}%. "
            f"Teaching Style: {row.get('teaching_style', 'Unknown')}. "
            f"Courses: {', '.join(row['courses_taught'][:5])}."
        )

    def _create_professor_display_chunk(self, row: pd.Series) -> str:
        """Create detailed display chunk for professor information."""
        display_parts = [
            f"👨‍🏫 Professor: {row['instructor']}",
            f"📊 Teaching Statistics:",
            f"  - Average GPA Given: {row['avg_gpa_given']:.2f}",
            f"  - A Rate: {row['avg_a_rate']:.1f}%",
            f"  - A+B Rate: {row.get('a_plus_b_rate', 0):.1f}%",
            f"  - Pass Rate: {row['avg_pass_rate']:.1f}%",
            f"  - DFW Rate: {row['avg_dfw_rate']:.1f}%",
            f"  - Total Students: {int(row['total_students'])}",
            f"  - Terms Taught: {row['terms_taught']}",
            f"  - Teaching Style: {row.get('teaching_style', 'Unknown')}"
        ]

        display_parts.append(f"\n📚 Courses Taught ({len(row['courses_taught'])}):")
        for i, course_code in enumerate(row['courses_taught'][:10], 1):
            title = self.get_course_title(course_code)
            display_parts.append(f"  {i}. {course_code}: {title}")

        if len(row['courses_taught']) > 10:
            display_parts.append(f"  ... and {len(row['courses_taught']) - 10} more courses")

        return "\n".join(display_parts)

    def _create_section_search_chunk(self, row: pd.Series) -> str:
        """Create search chunk for section information."""
        base_text = (
            f"Section: {row['course_code']} in {row['Term']} taught by {row['instructor']}. "
            f"Grades: A={row['A']}, B={row['B']}, C={row['C']}. "
            f"GPA: {row['gpa']:.2f}, Students: {row['Total Grades']}."
        )

        # Add career and grad information if available
        if 'Course Career' in row and row['Course Career'] and row['Course Career'] != "nan":
            base_text += f" Course Career: {row['Course Career']}."

        if 'GRAD' in row and row['GRAD'] and row['GRAD'] != "nan":
            base_text += f" GRAD: {row['GRAD']}."

        return base_text

    def _create_section_display_chunk(self, row: pd.Series) -> str:
        """Create display chunk for section information with all metadata."""
        display_parts = [
            f"📋 Section: {row['course_code']} in {row['Term']}",
            f"👨‍🏫 Instructor: {row['instructor']}",
            f"📊 Grades: A={row['A']}, B={row['B']}, C={row['C']}, D={row['D']}, F={row['F']}, W={row['W']}",
            f"🎯 Total Students: {row['Total Grades']}, GPA: {row['gpa']:.2f}"
        ]

        # Add career information if available
        if 'Course Career' in row and row['Course Career'] and row['Course Career'] != "nan":
            display_parts.append(f"🎯 Course Career: {row['Course Career']}")

        # Add GRAD information if available
        if 'GRAD' in row and row['GRAD'] and row['GRAD'] != "nan":
            display_parts.append(f"🎓 GRAD: {row['GRAD']}")

        return "\n".join(display_parts)

# ========================
# 6. VECTOR STORE MANAGER
# ========================
class VectorStoreManager:
    """Manages vector storage and retrieval with FAISS."""

    def __init__(self, config: DataConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.embed_model = None
        self.indices = {}

    def initialize_embedding_model(self, model_id: str) -> None:
        """Initialize the embedding model with error handling."""
        try:
            self.logger.info(f"Initializing embedding model: {model_id}")
            self.embed_model = SentenceTransformer(
                model_id,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.logger.info("Embedding model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise ModelLoadingError(f"Embedding model loading failed: {str(e)}")

    def build_or_load_index(self, df: pd.DataFrame, text_column: str,
                          index_name: str, force_rebuild: bool = False) -> faiss.Index:
        """Build or load FAISS index with caching."""
        index_path = f"{self.config.index_prefix}_{index_name}.index"

        if not force_rebuild and os.path.exists(index_path):
            self.logger.info(f"Loading existing index: {index_path}")
            try:
                index = faiss.read_index(index_path)
                self.indices[index_name] = index
                return index
            except Exception as e:
                self.logger.warning(f"Failed to load index, rebuilding: {str(e)}")

        self.logger.info(f"Building new index: {index_name}")
        try:
            corpus = df[text_column].tolist()
            # Disable multiprocessing to avoid segmentation faults on macOS
            embeddings = self.embed_model.encode(
                corpus,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=16,  # Reduced batch size for stability
                convert_to_numpy=True,
                device='cpu' if not torch.cuda.is_available() else 'cuda'
            )

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings.astype(np.float32))

            # Save index
            faiss.write_index(index, index_path)
            self.indices[index_name] = index

            self.logger.info(f"Index built and saved: {index_path}")
            return index

        except Exception as e:
            self.logger.error(f"Index building failed: {str(e)}")
            raise IndexBuildingError(f"Failed to build index {index_name}: {str(e)}")

    @lru_cache(maxsize=1000)
    def retrieve_similar(self, query: str, index_name: str, top_k: int = 3) -> List[int]:
        """Retrieve similar items with caching."""
        if index_name not in self.indices:
            raise IndexBuildingError(f"Index {index_name} not found")

        query_vec = self.embed_model.encode([query], normalize_embeddings=True)
        distances, indices = self.indices[index_name].search(query_vec.astype(np.float32), top_k)

        return indices[0].tolist()

# ========================
# 7. LLM MANAGER
# ========================
class LLMManager:
    """Manages LLM operations with optimizations."""

    def __init__(self, config: ModelConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.tokenizer = None
        self.model = None
        self.pipeline = None

    def initialize_models(self) -> None:
        """Initialize LLM models with optimizations."""
        try:
            self.logger.info(f"Loading tokenizer and model: {self.config.model_id}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device_map,
                trust_remote_code=True
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device_map
            )

            self.logger.info("LLM models initialized successfully")

        except Exception as e:
            self.logger.error(f"LLM initialization failed: {str(e)}")
            raise ModelLoadingError(f"LLM loading failed: {str(e)}")

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response with proper error handling."""
        try:
            start_time = time.time()

            generation_config = {
                'max_new_tokens': kwargs.get('max_new_tokens', self.config.max_new_tokens),
                'temperature': kwargs.get('temperature', self.config.temperature),
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id
            }

            result = self.pipeline(prompt, **generation_config)
            generated_text = result[0]['generated_text']

            # Extract only the new generated content
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()

            latency = time.time() - start_time
            self.logger.debug(f"Generation completed in {latency:.2f}s")

            return response

        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}")
            return "I apologize, but I encountered an error while generating a response."

