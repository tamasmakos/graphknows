### ENTITY KNOWLEDGE GRAPH


import os
import json
import networkx as nx
import re
import spacy
import asyncio
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
import warnings
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import logging
from difflib import SequenceMatcher
import numpy as np
from collections import defaultdict
import jellyfish  # For Levenshtein Distance

# Load environment variables from .env file

# LangChain imports  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# PyKEEN imports for coreference resolution and link prediction
try:
    import torch
    import pandas as pd
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    PYKEEN_AVAILABLE = True
except ImportError:
    PYKEEN_AVAILABLE = False

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# LangChain imports for relation extraction
try:
    from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False # Set first before logging

# Additional imports for integrated community detection  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import glob
from collections import Counter

# igraph and leidenalg for community detection
try:
    import igraph as ig
    import leidenalg as la
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False

# PyVis for HTML visualization
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# Sentence transformers for embeddings (RAG-ready)
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Visualization libraries for similarity heatmaps (already imported above)
VISUALIZATION_AVAILABLE = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if not LLM_AVAILABLE:
    logger.warning("LangChain dependencies not available. LLM-based relation extraction will be disabled.")

if not PYKEEN_AVAILABLE:
    logger.warning("PyKEEN not available. Advanced coreference resolution will be disabled.")

if not PYVIS_AVAILABLE:
    logger.warning("PyVis not available. HTML visualizations will be disabled.")

if not EMBEDDINGS_AVAILABLE:
    logger.warning("SentenceTransformers not available. RAG embeddings will be disabled.")

warnings.filterwarnings("ignore")

# Load spaCy model once at the beginning for efficiency
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

def _levenshtein_distance_safe(a: str, b: str) -> int:
    """Safe Levenshtein distance with fallback to SequenceMatcher-derived distance."""
    try:
        if hasattr(jellyfish, 'levenshtein_distance'):
            return int(jellyfish.levenshtein_distance(a, b))
    except Exception:
        pass
    try:
        ratio = SequenceMatcher(None, a, b).ratio()
        return int(round((1.0 - ratio) * max(len(a), len(b))))
    except Exception:
        # Worst-case fallback
        return abs(len(a) - len(b))

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Integrated Configuration (previously from kg_config.py)
DEFAULT_CONFIG = {
    # Data source configuration
    'input_dir': os.path.join(SCRIPT_DIR, '..', 'input', 'txt', 'translated'),
    'output_dir': os.path.join(SCRIPT_DIR, '..', 'output'),
    
    # Processing limits (for consistent comparison)
    'speech_limit': 100,
    'use_speech_limit': False,
    
    # Concurrency settings
    'max_concurrent_extractions': 8,
    
    # File patterns
    'file_pattern': '*.txt',
    'file_regex': r'CRE-(\d+)-(\d{4}-\d{2}-\d{2})-ITM-(\d{3})_EN\.txt',
    
    # Chunk configuration (for consistent chunking)
    'chunk_size': 3,  # sentences per chunk
    'use_overlapping_chunks': True,
    
    # Community detection parameters
    'community_detection': {
        'resolution_parameter': 1.0,
        'random_seed': 42,
        'iterations': 10,
        'min_comm_size': 2,
        'min_subcomm_size': 2,
        'sub_max_depth': 1,
        'sub_resolution_min': 0.7,
        'sub_resolution_max': 1.3,
        'sub_resolution_steps': 7,
        'sub_consistency_threshold': 0.75
    }
}

class KGConfig:
    """Configuration manager for Knowledge Graph generation"""
    
    def __init__(self, **overrides):
        """Initialize configuration with optional overrides"""
        self.config = DEFAULT_CONFIG.copy()
        
        # Apply any overrides
        for key, value in overrides.items():
            if key in self.config:
                self.config[key] = value
            else:
                # For nested configs like community_detection
                if isinstance(self.config.get(key.split('.')[0]), dict):
                    parent_key = key.split('.')[0]
                    child_key = '.'.join(key.split('.')[1:])
                    self.config[parent_key][child_key] = value
                else:
                    self.config[key] = value
    
    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set configuration value"""
        self.config[key] = value
    
    def to_dict(self):
        """Return configuration as dictionary"""
        return self.config.copy()
    
    def validate_paths(self):
        """Validate that input directory exists"""
        input_dir = self.get('input_dir')
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Create output directory if it doesn't exist
        output_dir = self.get('output_dir')
        os.makedirs(output_dir, exist_ok=True)
        
        return True

def get_shared_config(**overrides):
    """Get shared configuration instance"""
    config = KGConfig(**overrides)
    config.validate_paths()
    return config

def get_input_files(config):
    """Get list of input files to process"""
    input_dir = config.get('input_dir')
    file_pattern = config.get('file_pattern')
    
    # Get all .txt files matching the pattern
    pattern_path = os.path.join(input_dir, file_pattern)
    files = glob.glob(pattern_path)
    
    # Sort for consistent processing order
    files.sort()
    
    return files

def should_process_file(filename, config):
    """Check if file should be processed based on regex pattern"""
    file_regex = config.get('file_regex')
    if file_regex:
        pattern = re.compile(file_regex)
        return bool(pattern.match(os.path.basename(filename)))
    
    return filename.endswith('.txt')

# Load shared configuration (can be overridden by function parameters)
_DEFAULT_CONFIG = get_shared_config()

# Configuration
SPEECH_LIMIT = _DEFAULT_CONFIG.get('speech_limit')
USE_SPEECH_LIMIT = _DEFAULT_CONFIG.get('use_speech_limit')
MAX_CONCURRENT_EXTRACTIONS = _DEFAULT_CONFIG.get('max_concurrent_extractions')

# =============================================================================
# INTEGRATED TOPIC SUMMARIZATION (previously from topic_summarizer.py)
# =============================================================================

# Configuration for topic summarization concurrency
MAX_CONCURRENT_SUMMARIES = 10  # Adjust based on API rate limits

@dataclass
class SummarizationTask:
    """Task for generating title and summary for a topic/subtopic"""
    task_id: str
    community_id: int
    subcommunity_id: Optional[int]
    is_topic: bool
    concatenated_text: str
    chunk_ids: List[str]
    entity_ids: List[str]
    title: Optional[str] = None
    summary: Optional[str] = None

@dataclass
class SummarizerDependencies:
    """Shared dependencies for all summarizer agents"""
    graph: nx.DiGraph
    llm: ChatGroq
    graph_file_path: str
    output_file_path: str
    summarization_tasks: List[SummarizationTask] = None
    similarity_results: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.summarization_tasks is None:
            self.summarization_tasks = []
        if self.similarity_results is None:
            self.similarity_results = {}

@dataclass
class SimilarTopicPair:
    """Represents a pair of similar topics found by Levenshtein analysis"""
    topic1_id: str
    topic1_title: str
    topic1_level: str
    topic2_id: str
    topic2_title: str
    topic2_level: str
    similarity_score: float
    levenshtein_distance: int
    is_potential_duplicate: bool

# Initialize the Groq model for PydanticAI
# Removed unused GroqModel instance (PydanticAI not used)

# =============================================================================
# INTEGRATED COMMUNITY DETECTION (previously from community_detection.py)
# =============================================================================

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

@dataclass
class CommunityQualityMetrics:
    """Metrics for evaluating community quality and balance"""
    modularity: float
    size_variance: float
    avg_size: float
    min_size: int
    max_size: int
    size_ratio: float  # max_size / min_size
    text_length_variance: float
    optimal_for_summarization: bool

class CommunityDetector:
    """Community detection using Leiden algorithm."""
    
    def __init__(self):
        """Initialize the community detector."""
        pass
    
    def run_leiden_with_consistency(self, graph, resolution, n_runs=5):
        """Run Leiden algorithm multiple times and select most consistent result."""
        if graph.number_of_nodes() < 3 or graph.number_of_edges() == 0:
            return {node: 0 for node in graph.nodes()}
        
        if not IGRAPH_AVAILABLE:
            logger.warning("igraph and leidenalg not available, using simple community detection fallback")
            # Simple fallback: group nodes by degree
            degrees = dict(graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            communities = {}
            for node, degree in degrees.items():
                # Create communities based on degree bins
                community_id = min(int(degree / (max_degree / 4)), 3)  # 4 communities max
                communities[node] = community_id
            return communities
        
        # Convert to igraph
        node_list = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        g_ig = ig.Graph()
        g_ig.add_vertices(len(node_list))
        edge_list = [(node_to_idx[source], node_to_idx[target]) for source, target in graph.edges()]
        g_ig.add_edges(edge_list)
        
        # Run multiple times
        all_partitions = []
        community_counts = []
        
        for i in range(n_runs):
            try:
                partition_obj = la.find_partition(
                    g_ig, la.RBConfigurationVertexPartition,
                    resolution_parameter=resolution, seed=i
                )
                
                partition = {}
                for idx, community in enumerate(partition_obj):
                    for node_idx in community:
                        node_id = node_list[node_idx]
                        partition[node_id] = idx
                
                all_partitions.append(partition)
                community_counts.append(len(set(partition.values())))
                
            except Exception:
                continue
        
        if not all_partitions:
            return {node: 0 for node in graph.nodes()}
        
        # Select most common number of communities
        most_common_count = Counter(community_counts).most_common(1)[0][0]
        for i, partition in enumerate(all_partitions):
            if community_counts[i] == most_common_count:
                return partition
        
        return all_partitions[0]
    
    def optimize_resolution(self, graph):
        """Find optimal resolution parameter with comprehensive evaluation."""
        logger.info("Optimizing resolution parameter in range (0.3, 2.0) with 15 steps...")
        
        resolution_values = np.linspace(0.1, 3.0, 30)
        results = []
        
        for resolution in resolution_values:
            try:
                # Run multiple times for consistency evaluation
                logger.info("Evaluating community consistency across 3 runs...")
                consistency_results = []
                community_counts = []
                
                for run in range(7):
                    partition = self.run_leiden_with_consistency(graph, resolution, n_runs=1)
                    consistency_results.append(partition)
                    community_counts.append(len(set(partition.values())))
                
                # Calculate consistency score (prefer NMI, fallback to pairwise agreement)
                if len(consistency_results) >= 2:
                    consistency_score = self.calculate_partition_consistency_nmi(consistency_results)
                else:
                    consistency_score = 1.0
                
                logger.info(f"Community consistency score: {consistency_score:.3f}")
                logger.info(f"Community counts across runs: {community_counts}")
                
                # Use the first partition for modularity calculation
                partition = consistency_results[0]
                
                # Calculate modularity
                try:
                    community_sets = []
                    for comm_id in set(partition.values()):
                        community_nodes = set([node for node, comm in partition.items() if comm == comm_id])
                        community_nodes = community_nodes.intersection(set(graph.nodes()))
                        if community_nodes:
                            community_sets.append(community_nodes)
                    
                    modularity = nx.algorithms.community.modularity(graph, community_sets)
                except:
                    modularity = 0.0
                
                logger.info(f"Resolution {resolution:.2f}: {len(set(partition.values()))} communities, "
                           f"consistency={consistency_score:.3f}, modularity={modularity:.3f}")
                
                results.append({
                    'resolution': resolution,
                    'modularity': modularity,
                    'consistency': consistency_score,
                    'n_communities': len(set(partition.values())),
                    'partition': partition,
                    'score': consistency_score * 0.6 + modularity * 0.4  # Combined score
                })
                
            except Exception as e:
                logger.warning(f"Failed to evaluate resolution {resolution}: {e}")
                continue
        
        if not results:
            logger.warning("No valid results, using default resolution 1.0")
            return 1.0, self.run_leiden_with_consistency(graph, 1.0)
        
        # Select best based on combined score (consistency + modularity)
        best_result = max(results, key=lambda x: x['score'])
        
        logger.info(f"Best resolution: {best_result['resolution']:.2f} "
                   f"(consistency={best_result['consistency']:.3f}, "
                   f"modularity={best_result['modularity']:.3f})")
        
        return best_result['resolution'], best_result['partition']
    
    def calculate_partition_consistency(self, partitions):
        """Calculate consistency score between multiple partitions."""
        if len(partitions) < 2:
            return 1.0
        
        agreement_scores = []
        
        for i in range(len(partitions)):
            part_i = partitions[i]
            for j in range(i + 1, len(partitions)):
                part_j = partitions[j]
                
                # Get common nodes
                common_nodes = list(set(part_i.keys()) & set(part_j.keys()))
                
                if not common_nodes:
                    continue
                
                # Count node pairs that have the same community assignment
                same_assignment = 0
                total_pairs = 0
                
                for n1_idx in range(len(common_nodes)):
                    n1 = common_nodes[n1_idx]
                    for n2_idx in range(n1_idx + 1, len(common_nodes)):
                        n2 = common_nodes[n2_idx]
                        
                        # Check if nodes are in same community in both partitions
                        same_in_i = part_i[n1] == part_i[n2]
                        same_in_j = part_j[n1] == part_j[n2]
                        
                        if (same_in_i and same_in_j) or (not same_in_i and not same_in_j):
                            same_assignment += 1
                        
                        total_pairs += 1
                
                if total_pairs > 0:
                    agreement = same_assignment / total_pairs
                    agreement_scores.append(agreement)
        
        # Return average agreement
        return sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0

    def calculate_partition_consistency_nmi(self, partitions):
        """Calculate consistency using Normalized Mutual Information (NMI) when available.

        Falls back to calculate_partition_consistency if sklearn is not installed or an error occurs.
        """
        if len(partitions) < 2:
            return 1.0
        try:
            from sklearn.metrics import normalized_mutual_info_score
        except Exception:
            return self.calculate_partition_consistency(partitions)

        try:
            scores = []
            for i in range(len(partitions)):
                part_i = partitions[i]
                for j in range(i + 1, len(partitions)):
                    part_j = partitions[j]
                    # Intersect node sets
                    common_nodes = list(set(part_i.keys()) & set(part_j.keys()))
                    if not common_nodes:
                        continue
                    labels_i = [part_i[n] for n in common_nodes]
                    labels_j = [part_j[n] for n in common_nodes]
                    nmi = normalized_mutual_info_score(labels_i, labels_j)
                    scores.append(float(nmi))
            return float(sum(scores) / len(scores)) if scores else 0.0
        except Exception:
            return self.calculate_partition_consistency(partitions)
    
    def detect_communities(self, graph):
        """Main community detection method with comprehensive optimization."""
        logger.info("=== LEIDEN COMMUNITY DETECTION ===")
        logger.info("Running Leiden algorithm on entire graph (handles disconnected components automatically)")
        
        if graph.number_of_nodes() < 3 or graph.number_of_edges() == 0:
            logger.info("Graph too small for community detection")
            return {node: 0 for node in graph.nodes()}
        
        # Step 1: Comprehensive resolution optimization for entire graph
        logger.info("Graph has connected and potentially disconnected components")
        best_resolution, communities = self.optimize_resolution(graph)
        
        logger.info(f"Using optimized resolution {best_resolution:.2f}")
        
        # Log final community statistics
        community_counts = Counter(communities.values())
        community_sizes = list(community_counts.values())
        
        logger.info(f"Leiden algorithm found {len(set(communities.values()))} communities")
        logger.info(f"Community sizes: {dict(community_counts)}")
        
        return communities

    def _merge_small_communities(self, g: nx.Graph, partition: Dict[str, int], min_size: int) -> Dict[str, int]:
        """Merge communities smaller than min_size into the best neighboring community."""
        if min_size <= 1:
            return partition
        from collections import defaultdict as _dd
        comm_to_nodes: Dict[int, List[str]] = _dd(list)
        for n, cid in partition.items():
            comm_to_nodes[cid].append(n)
        assign = dict(partition)
        for cid, members in list(comm_to_nodes.items()):
            if len(members) >= min_size:
                continue
            # Count boundary edges to other communities
            neighbor_counts: Dict[int, int] = {}
            for n in members:
                for nbr in g.neighbors(n):
                    nid = partition.get(nbr)
                    if nid is None or nid == cid:
                        continue
                    neighbor_counts[nid] = neighbor_counts.get(nid, 0) + 1
            if not neighbor_counts:
                # fallback to largest existing community (other than cid)
                largest = None
                largest_size = -1
                for ocid, onodes in comm_to_nodes.items():
                    if ocid == cid:
                        continue
                    if len(onodes) > largest_size:
                        largest = ocid
                        largest_size = len(onodes)
                target = largest if largest is not None else cid
            else:
                target = max(neighbor_counts.items(), key=lambda x: x[1])[0]
            for n in members:
                assign[n] = target
        return assign

    def detect_subcommunities_leiden(
        self,
        entity_graph: nx.Graph,
        communities: Dict[str, int],
        min_sub_size: int = 2,
        sub_resolution_min: float = 0.7,
        sub_resolution_max: float = 1.3,
        sub_resolution_steps: int = 7,
        max_depth: int = 1
    ) -> Dict[str, Tuple[int, int]]:
        """Run Leiden inside each parent community to find meaningful subcommunities.

        Returns mapping node_id -> (parent_community_id, local_sub_id).
        """
        if max_depth <= 0:
            return {}
        node_to_sub: Dict[str, Tuple[int, int]] = {}
        # Group nodes by parent community
        by_comm: Dict[int, List[str]] = defaultdict(list)
        for n, cid in communities.items():
            by_comm[cid].append(n)

        gammas = list(np.linspace(sub_resolution_min, sub_resolution_max, max(2, sub_resolution_steps)))

        for comm_id, nodes in by_comm.items():
            if len(nodes) < max(2 * min_sub_size, 4):
                continue
            subg = entity_graph.subgraph(nodes).copy()
            # Baseline modularity: one cluster
            try:
                baseline_mod = nx.algorithms.community.modularity(subg, [set(nodes)])
            except Exception:
                baseline_mod = 0.0

            best_fixed: Optional[Dict[str, int]] = None
            best_mod = -1e9

            for gamma in gammas:
                part = self.run_leiden_with_consistency(subg, gamma, n_runs=3)
                if len(set(part.values())) < 2:
                    continue
                fixed = self._merge_small_communities(subg, part, min_sub_size)
                if len(set(fixed.values())) < 2:
                    continue
                comm_sets = []
                for sid in set(fixed.values()):
                    comm_sets.append({n for n, c in fixed.items() if c == sid})
                try:
                    mod = nx.algorithms.community.modularity(subg, comm_sets)
                except Exception:
                    mod = 0.0
                if mod > best_mod:
                    best_mod = mod
                    best_fixed = fixed

            if best_fixed is None:
                continue
            if best_mod <= baseline_mod:
                continue
            # Relabel local ids densely
            old_to_local: Dict[int, int] = {}
            next_local = 0
            for n in nodes:
                sid = best_fixed[n]
                if sid not in old_to_local:
                    old_to_local[sid] = next_local
                    next_local += 1
                node_to_sub[n] = (comm_id, old_to_local[sid])

        return node_to_sub

def extract_entity_relation_subgraph(graph):
    """Extract entity_relation subgraph enriched with speakers and SPEAKER_MENTIONS edges."""
    logger.info("Extracting entity_relation subgraph with speaker enrichment...")
    
    # Get all nodes with graph_type = "entity_relation" (entities/concepts)
    entity_nodes = [
        node_id for node_id, node_data in graph.nodes(data=True)
        if node_data.get('graph_type') == 'entity_relation'
    ]
    
    # Get all speaker nodes 
    speaker_nodes = [
        node_id for node_id, node_data in graph.nodes(data=True)
        if node_data.get('node_type') == 'SPEAKER'
    ]
    
    logger.info(f"Found {len(entity_nodes)} entity nodes and {len(speaker_nodes)} speaker nodes")
    
    # Create a new graph that will include both entities and speakers
    enriched_graph = nx.Graph()
    
    # Add all entity nodes and their existing relations
    entity_subgraph = graph.subgraph(entity_nodes)
    enriched_graph.add_nodes_from(entity_subgraph.nodes(data=True))
    enriched_graph.add_edges_from(entity_subgraph.edges(data=True))
    
    # Add speaker nodes to the enriched graph
    for speaker_id in speaker_nodes:
        speaker_data = graph.nodes[speaker_id].copy()
        # Keep speakers as part of entity_relation graph for community detection
        speaker_data['graph_type'] = 'entity_relation'
        speaker_data['node_type'] = 'SPEAKER'  # Keep original type for identification
        enriched_graph.add_node(speaker_id, **speaker_data)
    
    # Skip adding SPEAKER_MENTIONS edges; speakers remain in graph but not linked for embeddings/communities
    existing_speaker_mentions = 0
    
    # Do not synthesize SPEAKER_MENTIONS edges; keep speakers unconnected for embeddings/communities
    logger.info("Skipping SPEAKER_MENTIONS edge creation for embeddings/community isolation")
    
    # Count total connections
    total_speaker_entity_edges = 0
    
    logger.info(f"Enriched graph has {enriched_graph.number_of_nodes()} nodes and {enriched_graph.number_of_edges()} edges")
    logger.info(f"  - {len(entity_nodes)} entities with {entity_subgraph.number_of_edges()} entity-entity relations")
    logger.info(f"  - {len(speaker_nodes)} speakers with {total_speaker_entity_edges} speaker-entity connections")
    
    return enriched_graph

def detect_balanced_subcommunities(entity_graph: nx.Graph, communities: Dict[str, int], 
                                  target_subcomm_size: int = 20) -> Dict[str, int]:
    """Detect subcommunities with size balancing for optimal summarization"""
    
    logger.info("Detecting balanced subcommunities...")
    subcommunities = {}
    global_subcommunity_id = 0
    
    # Group nodes by community
    communities_by_id = defaultdict(list)
    for node, community_id in communities.items():
        communities_by_id[community_id].append(node)
    
    for community_id, community_nodes in communities_by_id.items():
        comm_size = len(community_nodes)
        
        if comm_size <= target_subcomm_size:
            # Small community - keep as single subcommunity
            for node in community_nodes:
                subcommunities[node] = global_subcommunity_id
            global_subcommunity_id += 1
            
        elif comm_size <= target_subcomm_size * 2:
            # Medium community - split into 2 balanced subcommunities
            mid_point = comm_size // 2
            for i, node in enumerate(community_nodes):
                subcomm_id = global_subcommunity_id + (i // mid_point)
                subcommunities[node] = subcomm_id
            global_subcommunity_id += 2
            
        else:
            # Large community - use more sophisticated splitting
            num_subcommunities = max(2, comm_size // target_subcomm_size)
            
            # Create subgraph for this community
            subgraph = entity_graph.subgraph(community_nodes).copy()
            
            if subgraph.number_of_edges() > 0:
                # Use graph-based clustering for better splits
                try:
                    subcomm_assignments = split_community_with_graph_clustering(
                        subgraph, num_subcommunities, community_nodes
                    )
                    for node, local_subcomm_id in subcomm_assignments.items():
                        subcommunities[node] = global_subcommunity_id + local_subcomm_id
                    global_subcommunity_id += num_subcommunities
                    
                except Exception as e:
                    logger.warning(f"Graph clustering failed for community {community_id}, using simple split: {e}")
                    # Fallback to simple split
                    for i, node in enumerate(community_nodes):
                        subcomm_id = global_subcommunity_id + (i % num_subcommunities)
                        subcommunities[node] = subcomm_id
                    global_subcommunity_id += num_subcommunities
            else:
                # No edges - simple split
                for i, node in enumerate(community_nodes):
                    subcomm_id = global_subcommunity_id + (i % num_subcommunities)
                    subcommunities[node] = subcomm_id
                global_subcommunity_id += num_subcommunities
    
    logger.info(f"Created {global_subcommunity_id} balanced subcommunities")
    return subcommunities

def split_community_with_graph_clustering(subgraph: nx.Graph, num_clusters: int, 
                                        node_list: List[str]) -> Dict[str, int]:
    """Split a community using graph-aware clustering"""
    
    if subgraph.number_of_nodes() < num_clusters:
        # More clusters than nodes - each node gets its own cluster
        return {node: i for i, node in enumerate(node_list)}
    
    if subgraph.number_of_edges() == 0:
        # No connections - simple round-robin assignment
        return {node: i % num_clusters for i, node in enumerate(node_list)}
    
    try:
        # Use spectral clustering based on graph structure
        adjacency = nx.adjacency_matrix(subgraph)
        
        # Convert to dense for small graphs
        if adjacency.shape[0] < 1000:
            adjacency = adjacency.todense()
        
        # Simple k-means on adjacency matrix rows
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(adjacency)
        
        node_to_cluster = {}
        for i, node in enumerate(subgraph.nodes()):
            node_to_cluster[node] = cluster_labels[i]
        
        return node_to_cluster
        
    except Exception as e:
        logger.warning(f"Spectral clustering failed: {e}, using simple assignment")
        # Fallback to simple assignment
        return {node: i % num_clusters for i, node in enumerate(node_list)}

def add_enhanced_community_attributes_to_graph(graph: nx.DiGraph, communities: Dict[str, int], 
                                             subcommunities: Dict[str, Tuple[int, int]]) -> nx.DiGraph:
    """Enhanced version that creates proper hierarchical connections: Entities→Subcommunities→Parent Communities
    Expects subcommunities mapping: entity_id -> (parent_community_id, local_sub_id).
    """
    logger.info("Creating PROPER community hierarchy: Entities→Subcommunities→Parent Communities...")
    
    # Create ParentCommunity nodes  
    community_nodes_created = 0
    unique_communities = set(communities.values())
    for comm_id in unique_communities:
        community_node_id = f"COMMUNITY_{comm_id}"
        if community_node_id not in graph:
            graph.add_node(community_node_id,
                          node_type="COMMUNITY", 
                          graph_type="community",
                          community_id=comm_id)
            community_nodes_created += 1
    
    # Create SubCommunity nodes and edges
    subcommunity_nodes_created = 0
    in_community_edges_created = 0
    parent_community_edges_created = 0
    created_sub_nodes = set()
    
    for node_id, pair in subcommunities.items():
        if node_id not in graph:
            continue
        parent_comm_id, local_sub_id = pair
        sub_node_id = f"SUBCOMMUNITY_{parent_comm_id}_{local_sub_id}"
        if sub_node_id not in graph:
            graph.add_node(sub_node_id,
                          node_type="SUBCOMMUNITY",
                          graph_type="community",
                          community_id=parent_comm_id,
                          subcommunity_local_id=local_sub_id)
            subcommunity_nodes_created += 1
        created_sub_nodes.add(sub_node_id)
        # Entity -> Subcommunity
        if not graph.has_edge(node_id, sub_node_id):
            graph.add_edge(node_id, sub_node_id,
                          label="IN_COMMUNITY",
                          graph_type="community")
            in_community_edges_created += 1
        # Subcommunity -> Parent community
        community_node_id = f"COMMUNITY_{parent_comm_id}"
        if not graph.has_edge(sub_node_id, community_node_id):
            graph.add_edge(sub_node_id, community_node_id,
                          label="PARENT_COMMUNITY",
                          graph_type="community")
            parent_community_edges_created += 1
    
    logger.info(f"Created {subcommunity_nodes_created} SubCommunity nodes")
    logger.info(f"Created {community_nodes_created} ParentCommunity nodes") 
    logger.info(f"Created {in_community_edges_created} entity→subcommunity IN_COMMUNITY relationships")
    logger.info(f"Created {parent_community_edges_created} subcommunity→parent PARENT_COMMUNITY relationships")
    logger.info("✅ PROPER HIERARCHY: Entities→Subcommunities→Parent Communities")
    
    return graph

def evaluate_community_quality(graph: nx.DiGraph, communities: Dict[str, int], 
                             target_summary_length: int = 50000) -> CommunityQualityMetrics:
    """Evaluate community quality for summarization purposes"""
    
    def calculate_text_length_for_community(graph: nx.DiGraph, entity_ids: List[str]) -> int:
        """Calculate total text length for entities in a community"""
        total_chars = 0
        
        for entity_id in entity_ids:
            # Find chunks connected to this entity
            for predecessor in graph.predecessors(entity_id):
                edge_data = graph.get_edge_data(predecessor, entity_id)
                if (edge_data and 
                    edge_data.get('label') == 'HAS_ENTITY' and
                    graph.nodes[predecessor].get('node_type') == 'CHUNK'):
                    
                    # Get text from sentences attribute
                    chunk_data = graph.nodes[predecessor]
                    if 'sentences' in chunk_data:
                        sentences = chunk_data['sentences']
                        if isinstance(sentences, list):
                            chunk_text = ' '.join(sentences)
                        else:
                            chunk_text = str(sentences)
                        total_chars += len(chunk_text)
        
        return total_chars
    
    # Group entities by community
    community_sizes = defaultdict(int)
    community_text_lengths = defaultdict(int)
    
    for entity_id, comm_id in communities.items():
        community_sizes[comm_id] += 1
        # Calculate text length for this entity's chunks
        text_length = calculate_text_length_for_community(graph, [entity_id])
        community_text_lengths[comm_id] += text_length
    
    sizes = list(community_sizes.values())
    text_lengths = list(community_text_lengths.values())
    
    if not sizes:
        return CommunityQualityMetrics(0, 0, 0, 0, 0, 1, 0, False)
    
    # Calculate basic metrics
    avg_size = np.mean(sizes)
    size_variance = np.var(sizes) if len(sizes) > 1 else 0
    min_size = min(sizes)
    max_size = max(sizes)
    size_ratio = max_size / min_size if min_size > 0 else float('inf')
    
    text_length_variance = np.var(text_lengths) if len(text_lengths) > 1 else 0
    
    # Check if communities are optimal for summarization
    # Ideal: 10K-100K characters per community, not too imbalanced
    optimal_communities = sum(1 for length in text_lengths 
                            if 10000 <= length <= 100000)
    total_communities = len(text_lengths)
    optimal_ratio = optimal_communities / total_communities if total_communities > 0 else 0
    
    optimal_for_summarization = (
        optimal_ratio >= 0.7 and  # At least 70% in optimal range
        size_ratio <= 10 and     # Not too imbalanced
        max(text_lengths) <= 200000  # No extremely large communities
    )
    
    # Calculate modularity (simplified approximation)
    modularity = calculate_simple_modularity(graph, communities)
    
    return CommunityQualityMetrics(
        modularity=modularity,
        size_variance=size_variance,
        avg_size=avg_size,
        min_size=min_size,
        max_size=max_size,
        size_ratio=size_ratio,
        text_length_variance=text_length_variance,
        optimal_for_summarization=optimal_for_summarization
    )

def calculate_simple_modularity(graph: nx.DiGraph, communities: Dict[str, int]) -> float:
    """Calculate modularity for the entity subgraph using NetworkX's implementation.

    This replaces the prior ratio-of-internal-edges approximation with a proper modularity score.
    """
    try:
        # Extract entity subgraph and convert to undirected for modularity
        entity_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'ENTITY_CONCEPT']
        if not entity_nodes:
            return 0.0
        undirected_graph = graph.subgraph(entity_nodes).to_undirected()
        if undirected_graph.number_of_edges() == 0:
            return 0.0

        # Build community sets aligned to nodes actually present in the subgraph
        community_sets: List[Set[str]] = []
        for comm_id in set(communities.values()):
            nodes_in_comm = {n for n, cid in communities.items() if cid == comm_id and undirected_graph.has_node(n)}
            if nodes_in_comm:
                community_sets.append(nodes_in_comm)
        if not community_sets:
            return 0.0

        return float(nx.algorithms.community.modularity(undirected_graph, community_sets))
    except Exception as e:
        logger.warning(f"Could not calculate modularity: {e}")
        return 0.0


def calculate_entity_relation_centrality_measures(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculate comprehensive centrality measures for all entity-relation nodes.
    
    This function calculates multiple centrality measures for nodes with graph_type = "entity_relation",
    excluding community, lexical graph, speaker, and speech nodes.
    
    Args:
        graph: NetworkX DiGraph containing the knowledge graph
        
    Returns:
        Dictionary containing centrality statistics and results
    """
    logger.info("=== CALCULATING CENTRALITY MEASURES FOR ENTITY-RELATION NODES ===")
    
    # Get all entity-relation nodes (entities/concepts)
    entity_relation_nodes = [
        node_id for node_id, node_data in graph.nodes(data=True)
        if node_data.get('graph_type') == 'entity_relation'
    ]
    
    logger.info(f"Found {len(entity_relation_nodes)} entity-relation nodes for centrality calculation")
    
    if len(entity_relation_nodes) < 2:
        logger.warning("Not enough entity-relation nodes for centrality calculation (need at least 2)")
        return {
            'nodes_processed': len(entity_relation_nodes),
            'centrality_measures': {},
            'statistics': {},
            'error': 'Insufficient nodes for centrality calculation'
        }
    
    # Create subgraph with only entity-relation nodes and their connections
    entity_subgraph = graph.subgraph(entity_relation_nodes).copy()
    
    # Convert to undirected for centrality calculations (most measures work on undirected graphs)
    undirected_graph = entity_subgraph.to_undirected()
    
    if undirected_graph.number_of_edges() == 0:
        logger.warning("No edges found in entity-relation subgraph")
        return {
            'nodes_processed': len(entity_relation_nodes),
            'centrality_measures': {},
            'statistics': {},
            'error': 'No edges in entity-relation subgraph'
        }
    
    logger.info(f"Entity-relation subgraph: {undirected_graph.number_of_nodes()} nodes, {undirected_graph.number_of_edges()} edges")
    
    centrality_results = {}
    statistics = {}
    
    try:
        # 1. Degree Centrality
        logger.info("Calculating degree centrality...")
        degree_centrality = nx.degree_centrality(undirected_graph)
        centrality_results['degree'] = degree_centrality
        
        # 2. Betweenness Centrality
        logger.info("Calculating betweenness centrality...")
        betweenness_centrality = nx.betweenness_centrality(undirected_graph, normalized=True)
        centrality_results['betweenness'] = betweenness_centrality
        
        # 3. Closeness Centrality
        logger.info("Calculating closeness centrality...")
        closeness_centrality = nx.closeness_centrality(undirected_graph)
        centrality_results['closeness'] = closeness_centrality
        
        # 4. Eigenvector Centrality
        logger.info("Calculating eigenvector centrality...")
        try:
            eigenvector_centrality = nx.eigenvector_centrality(undirected_graph, max_iter=1000, tol=1e-06)
            centrality_results['eigenvector'] = eigenvector_centrality
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality failed to converge, using Katz centrality as fallback")
            try:
                katz_centrality = nx.katz_centrality(undirected_graph, max_iter=1000, tol=1e-06)
                centrality_results['katz'] = katz_centrality
            except nx.PowerIterationFailedConvergence:
                logger.warning("Katz centrality also failed to converge, skipping eigenvector-like measures")
        
        # 5. PageRank Centrality
        logger.info("Calculating PageRank centrality...")
        pagerank_centrality = nx.pagerank(undirected_graph, alpha=0.85, max_iter=1000, tol=1e-06)
        centrality_results['pagerank'] = pagerank_centrality
        
        # 6. Harmonic Centrality
        logger.info("Calculating harmonic centrality...")
        harmonic_centrality = nx.harmonic_centrality(undirected_graph)
        centrality_results['harmonic'] = harmonic_centrality
        
        # 7. Load Centrality (alternative to betweenness)
        logger.info("Calculating load centrality...")
        try:
            load_centrality = nx.load_centrality(undirected_graph)
            centrality_results['load'] = load_centrality
        except Exception as e:
            logger.warning(f"Load centrality calculation failed: {e}")
        
        # 8. Current Flow Betweenness (for weighted graphs)
        logger.info("Calculating current flow betweenness centrality...")
        try:
            current_flow_betweenness = nx.current_flow_betweenness_centrality(undirected_graph)
            centrality_results['current_flow_betweenness'] = current_flow_betweenness
        except Exception as e:
            logger.warning(f"Current flow betweenness calculation failed: {e}")
        
        # 9. Current Flow Closeness
        logger.info("Calculating current flow closeness centrality...")
        try:
            current_flow_closeness = nx.current_flow_closeness_centrality(undirected_graph)
            centrality_results['current_flow_closeness'] = current_flow_closeness
        except Exception as e:
            logger.warning(f"Current flow closeness calculation failed: {e}")
        
        # Add centrality measures to graph nodes
        logger.info("Adding centrality measures to graph nodes...")
        for node_id in entity_relation_nodes:
            node_data = graph.nodes[node_id]
            
            # Add each centrality measure to the node
            for measure_name, measure_scores in centrality_results.items():
                if node_id in measure_scores:
                    node_data[f'{measure_name}_centrality'] = float(measure_scores[node_id])
        
        # Calculate statistics for each centrality measure
        logger.info("Calculating centrality statistics...")
        for measure_name, measure_scores in centrality_results.items():
            if measure_scores:
                scores = list(measure_scores.values())
                statistics[measure_name] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'median': float(np.median(scores)),
                    'q25': float(np.percentile(scores, 25)),
                    'q75': float(np.percentile(scores, 75))
                }
        
        # Find top nodes for each centrality measure
        top_nodes = {}
        for measure_name, measure_scores in centrality_results.items():
            if measure_scores:
                sorted_nodes = sorted(measure_scores.items(), key=lambda x: x[1], reverse=True)
                top_nodes[measure_name] = sorted_nodes[:10]  # Top 10 nodes
        
        logger.info("✅ Centrality measures calculated successfully")
        logger.info(f"Processed {len(entity_relation_nodes)} entity-relation nodes")
        logger.info(f"Calculated {len(centrality_results)} centrality measures")
        
        return {
            'nodes_processed': len(entity_relation_nodes),
            'centrality_measures': centrality_results,
            'statistics': statistics,
            'top_nodes': top_nodes,
            'graph_info': {
                'nodes': undirected_graph.number_of_nodes(),
                'edges': undirected_graph.number_of_edges(),
                'density': nx.density(undirected_graph),
                'is_connected': nx.is_connected(undirected_graph),
                'num_components': nx.number_connected_components(undirected_graph)
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating centrality measures: {e}")
        return {
            'nodes_processed': len(entity_relation_nodes),
            'centrality_measures': {},
            'statistics': {},
            'error': str(e)
        }

def filter_embedding_attributes_from_graph(graph: nx.Graph) -> Dict[str, Any]:
    """
    Convert graph to node-link data format while filtering out embedding-related attributes.
    
    This function removes all embedding-related attributes from nodes to prevent them
    from being included in JSON output files.
    
    Args:
        graph: NetworkX graph to convert
        
    Returns:
        Dictionary in node-link data format without embedding attributes
    """
    # Get the standard node-link data
    graph_data = nx.node_link_data(graph, edges="links")
    
    # Define embedding-related attribute names to filter out
    embedding_attributes = {
        'kge_embedding', 'embedding', 'embeddings', 'vector', 'vectors',
        'embedding_vector', 'node_embedding', 'entity_embedding',
        'semantic_embedding', 'graph_embedding', 'feature_vector'
    }
    
    # Filter out embedding attributes from nodes
    if 'nodes' in graph_data:
        filtered_nodes = []
        for node in graph_data['nodes']:
            filtered_node = {}
            for key, value in node.items():
                # Skip embedding-related attributes
                if key.lower() not in embedding_attributes and not key.lower().endswith('_embedding'):
                    filtered_node[key] = value
            filtered_nodes.append(filtered_node)
        graph_data['nodes'] = filtered_nodes
    
    return graph_data

def create_output_directory():
    """Create output directory for plots."""
    output_plots_dir = "/workspaces/kg/output"
    if not os.path.exists(output_plots_dir):
        os.makedirs(output_plots_dir)
        logger.info(f"Created output directory: {output_plots_dir}")
    return output_plots_dir


def save_and_plot_top_centrality_measures(centrality_results: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    Save and plot top 10 entities for each centrality measure.
    
    Args:
        centrality_results: Results from calculate_entity_relation_centrality_measures
        output_dir: Output directory for saving files
        
    Returns:
        Dictionary with file paths for saved plots and data
    """
    logger.info("Saving and plotting top centrality measures...")
    
    output_files = {}
    top_nodes = centrality_results.get('top_nodes', {})
    
    if not top_nodes:
        logger.warning("No top nodes data available for plotting")
        return output_files
    
    try:
        import matplotlib.pyplot as plt
        import json
        
        # Create centrality plots directory
        centrality_dir = os.path.join(output_dir, "centrality_analysis")
        os.makedirs(centrality_dir, exist_ok=True)
        
        # Save top nodes data as JSON
        top_nodes_file = os.path.join(centrality_dir, "top_centrality_nodes.json")
        with open(top_nodes_file, 'w') as f:
            json.dump(top_nodes, f, indent=2)
        output_files['top_nodes_json'] = top_nodes_file
        
        # Create plots for each centrality measure
        for measure_name, top_entities in top_nodes.items():
            if not top_entities:
                continue
                
            # Extract data for plotting
            entity_names = [entity[0] for entity in top_entities]
            scores = [entity[1] for entity in top_entities]
            
            # Create horizontal bar plot
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(entity_names)), scores)
            plt.yticks(range(len(entity_names)), entity_names)
            plt.xlabel(f'{measure_name.replace("_", " ").title()} Score')
            plt.title(f'Top 10 Entities by {measure_name.replace("_", " ").title()}')
            plt.gca().invert_yaxis()  # Highest scores at top
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, scores)):
                plt.text(bar.get_width() + max(scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{score:.4f}', va='center', ha='left')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(centrality_dir, f"top10_{measure_name}_centrality.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            output_files[f'{measure_name}_plot'] = plot_file
            logger.info(f"Saved {measure_name} centrality plot: {plot_file}")
        
        logger.info(f"✅ Saved {len(output_files)} centrality analysis files to {centrality_dir}")
        
    except ImportError as e:
        logger.warning(f"Matplotlib not available for plotting: {e}")
    except Exception as e:
        logger.error(f"Error creating centrality plots: {e}")
    
    return output_files

# =============================================================================
# PYVIS VISUALIZATION & RAG EMBEDDINGS
# =============================================================================

def generate_pyvis_visualization(
    graph: nx.DiGraph, 
    output_path: str,
    title: str = "Knowledge Graph",
    communities: Dict[str, int] = None,
    max_nodes: int = 500
) -> str:
    """Generate interactive PyVis HTML visualization using the same style as inverted_index.GraphVisualizer"""

    if not PYVIS_AVAILABLE:
        logger.warning("PyVis not available. Skipping HTML visualization.")
        return None

    logger.info(f"Generating PyVis visualization (inverted_index style): {title}")

    # Build an entity-relation-only subgraph: only ENTITY_CONCEPT nodes and entity_relation edges
    entity_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'ENTITY_CONCEPT']
    viz_graph = graph.subgraph(entity_nodes).copy()
    # Optional sampling to limit node count
    if viz_graph.number_of_nodes() > max_nodes:
        logger.info(f"Entity graph has {viz_graph.number_of_nodes()} nodes, sampling {max_nodes} for visualization")
        sampled_nodes = list(viz_graph.nodes())[:max_nodes]
        viz_graph = viz_graph.subgraph(sampled_nodes).copy()

    # Create PyVis network (white background, black font, no explicit directed flag)
    net = Network(height="900px", width="100%", notebook=False, bgcolor="#ffffff", font_color="black")
    net.show_buttons(filter_=["physics"])

    # Prepare community colors
    community_colors = {}
    unique_communities = set()
    if communities:
        unique_communities = set(communities.values())
    else:
        # Try detect from node attribute 'community_id'
        for node in viz_graph.nodes():
            comm_id = viz_graph.nodes[node].get('community_id')
            if comm_id is not None:
                unique_communities.add(comm_id)
    if unique_communities:
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57',
                  '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43',
                  '#ff7675', '#74b9ff', '#a29bfe', '#fd79a8', '#fdcb6e',
                  '#6c5ce7', '#00b894', '#e17055', '#636e72', '#2d3436']
        for i, comm_id in enumerate(sorted(unique_communities, key=str)):
            community_colors[comm_id] = colors[i % len(colors)]

    # Add nodes (entity nodes only)
    added_nodes = set()
    for node in viz_graph.nodes():
        ndata = viz_graph.nodes[node]

        # Build hover title from all attributes, formatting common centralities
        title_parts = []
        # Surface summaries prominently if present
        if 'community_summary' in ndata and ndata.get('community_summary'):
            s = ndata.get('community_summary')
            if isinstance(s, str) and len(s) > 400:
                s = s[:400] + "..."
            title_parts.append(f"SUMMARY: {s}")
        if 'summary' in ndata and ndata.get('summary'):
            s2 = ndata.get('summary')
            if isinstance(s2, str) and len(s2) > 400:
                s2 = s2[:400] + "..."
            title_parts.append(f"SUMMARY: {s2}")
        for k, v in ndata.items():
            if k in ['pagerank', 'closeness', 'betweenness'] and isinstance(v, (int, float)):
                title_parts.append(f"{k}: {v:.4f}")
            else:
                title_parts.append(f"{k}: {v}")
        node_title = "\n".join(title_parts)

        # Color and size based solely on communities for entity nodes
        comm_id = None
        if communities and node in communities:
            comm_id = communities[node]
        else:
            comm_id = ndata.get('community_id')
        color = community_colors.get(comm_id, "#636e72")
        size = 10

        node_str = str(node)
        net.add_node(node_str, label=node_str, title=node_title, size=size, color=color)
        added_nodes.add(node_str)

    # Add only entity_relation edges with styling similar to inverted_index.GraphVisualizer
    for u, v, data in viz_graph.edges(data=True):
        if data.get("graph_type") != "entity_relation":
            continue
        edge_title = "\n".join([f"{k}: {val}" for k, val in data.items()])
        edge_type = data.get("edge_type", "")
        # Visible edge label text (prefer 'label', then 'relation_type')
        label_text = data.get("label") or data.get("relation_type") or ""
        if not edge_type:
            # Map from label to edge_type for styling parity
            label = data.get("label", "")
            if label in ("HAS_CHUNK",):
                edge_type = "HAS_CHUNK"
            elif label in ("IN_COMMUNITY",):
                edge_type = "IN_COMMUNITY"
            elif label in ("HAS_SUBCOMMUNITY", "PARENT_COMMUNITY"):
                edge_type = "HAS_SUBCOMMUNITY"
            elif label in ("HAS_ENTITY", "SPEAKER_MENTIONS", "ENTITY_RELATION"):
                edge_type = label  # fall through to default styling

        u_str, v_str = str(u), str(v)
        # Only add edges if both endpoints are present in the visualization
        if u_str in added_nodes and v_str in added_nodes:
            if edge_type == "HAS_CHUNK":
                net.add_edge(u_str, v_str, title=edge_title, color="#3498db", width=3, arrows="to", label=label_text)
            elif edge_type == "HAS_KEYWORD":
                net.add_edge(u_str, v_str, title=edge_title, color="#9b59b6", width=2, arrows="to", label=label_text)
            elif edge_type == "IN_COMMUNITY":
                net.add_edge(u_str, v_str, title=edge_title, color="#8e44ad", width=2, arrows="to", label=label_text)
            elif edge_type == "HAS_SUBCOMMUNITY":
                net.add_edge(u_str, v_str, title=edge_title, color="#9b59b6", width=2, arrows="to", label=label_text)
            elif data.get("similarity_edge"):
                net.add_edge(u_str, v_str, title=edge_title, color="#3498db", width=2, arrows="to", dashes=True, label=label_text)
            elif data.get("dep"):
                net.add_edge(u_str, v_str, title=edge_title, color="#e67e22", width=1, arrows="to", label=label_text)
            else:
                net.add_edge(u_str, v_str, title=edge_title, color="#95a5a6", arrows="to", label=label_text)

    net.save_graph(output_path)
    logger.info(f"PyVis visualization saved to {output_path}")
    return output_path

def generate_rag_embeddings(
    graph: nx.DiGraph,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> Dict[str, np.ndarray]:
    """Generate embeddings for important nodes for RAG applications"""
    
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("SentenceTransformers not available. Skipping embedding generation.")
        return {}
    
    logger.info("Generating RAG embeddings for important nodes...")
    
    # Load embedding model
    model = SentenceTransformer(embedding_model)
    
    embeddings = {}
    
    # Target nodes for embedding generation (exclude SPEAKER)
    target_node_types = ['ENTITY_CONCEPT', 'COMMUNITY', 'SUBCOMMUNITY']
    
    texts_to_embed = []
    node_ids = []
    
    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get('node_type')
        
        if node_type in target_node_types:
            # Create text for embedding
            text_parts = []
            
            # Add name/title
            if 'name' in node_data:
                text_parts.append(node_data['name'])
            elif 'title' in node_data:
                text_parts.append(node_data['title'])
            
            # Add summary if available
            if 'summary' in node_data:
                text_parts.append(node_data['summary'])
            
            # Add entity type context
            if node_type == 'ENTITY_CONCEPT' and 'entity_type' in node_data:
                text_parts.append(f"This is a {node_data['entity_type']}")
            
            # For communities, add context about contained entities
            if node_type in ['COMMUNITY', 'SUBCOMMUNITY']:
                # Find entities in this community
                related_entities = []
                for pred in graph.predecessors(node_id):
                    pred_data = graph.nodes[pred]
                    if pred_data.get('node_type') == 'ENTITY_CONCEPT':
                        entity_name = pred_data.get('name', pred)
                        related_entities.append(entity_name)
                
                if related_entities:
                    text_parts.append(f"Contains entities: {', '.join(related_entities[:10])}")
            
            # Create combined text
            if text_parts:
                combined_text = ' '.join(text_parts)
                texts_to_embed.append(combined_text)
                node_ids.append(node_id)
    
    if texts_to_embed:
        logger.info(f"Generating embeddings for {len(texts_to_embed)} nodes...")
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            batch_embeddings = model.encode(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # Store embeddings
        for node_id, embedding in zip(node_ids, all_embeddings):
            embeddings[node_id] = embedding
            
            # Also add embedding to the graph node data
            graph.nodes[node_id]['embedding'] = embedding.tolist()  # Convert to list for JSON serialization
        
        logger.info(f"Generated embeddings for {len(embeddings)} nodes")
    
    return embeddings

# def save_embeddings_for_rag(
#     embeddings: Dict[str, np.ndarray],
#     output_path: str
# ):
#     """Save embeddings in a format suitable for RAG applications"""
#     
#     # Convert numpy arrays to lists for JSON serialization
#     embeddings_dict = {
#         node_id: embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
#         for node_id, embedding in embeddings.items()
#     }
#     
#     rag_data = {
#         "embeddings": embeddings_dict,
#         "embedding_dim": len(next(iter(embeddings_dict.values()))) if embeddings_dict else 0,
#         "model": "all-MiniLM-L6-v2",
#         "timestamp": datetime.now().isoformat(),
#         "total_embeddings": len(embeddings_dict)
#     }
#     
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(rag_data, f, indent=2, ensure_ascii=False)
#     
#     logger.info(f"RAG embeddings saved to {output_path}")

# =============================================================================
# GLOBAL KGE TRAINING (single shared space for all entities)
# =============================================================================

def _collect_global_entity_relation_triples(graph: nx.DiGraph) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    for u, v, data in graph.edges(data=True):
        if data.get('graph_type') == 'entity_relation':
            rel = data.get('relation_type') or data.get('label') or 'RELATED_TO'
            triples.append((str(u), str(rel), str(v)))
    return triples

def _train_and_cache_global_kge(
    graph: nx.DiGraph,
    output_dir: str,
    model_name: str = 'DistMult',
    embedding_dim: int = 64,
    lr: float = 1e-2,
    num_epochs: int = 50,
    patience: int = 5
) -> Dict[str, List[float]]:
    """Train a single KGE model over the full entity-relation graph and cache embeddings.

    Attaches vectors to nodes as `kge_embedding` and also populates GLOBAL_ENTITY_EMBEDDINGS.
    """
    global GLOBAL_ENTITY_EMBEDDINGS
    if not PYKEEN_AVAILABLE:
        logger.warning("PyKEEN not available; skipping global KGE training.")
        return {}

    triples = _collect_global_entity_relation_triples(graph)
    if not triples:
        logger.info("No entity-relation triples available for global KGE training.")
        return {}

    try:
        import pandas as _pd
        df = _pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
        tf = TriplesFactory.from_labeled_triples(df[['head', 'relation', 'tail']].values)

        result = pipeline(
            model=model_name,
            training=tf,
            validation=tf,
            testing=tf,
            model_kwargs=dict(embedding_dim=embedding_dim),
            optimizer_kwargs=dict(lr=lr),
            training_kwargs=dict(num_epochs=num_epochs, use_tqdm_batch=False, checkpoint_frequency=0),
            evaluation_kwargs=dict(use_tqdm=False),
            stopper='early',
            stopper_kwargs=dict(frequency=2, patience=patience),
            random_seed=42
        )

        # Extract entity embeddings
        entity_to_id = tf.entity_to_id
        if not entity_to_id:
            return {}
        all_ids = torch.arange(len(entity_to_id))
        with torch.no_grad():
            embs = result.model.entity_representations[0](all_ids).cpu().numpy()
        id_to_entity = {idx: ent for ent, idx in entity_to_id.items()}
        embeddings = {id_to_entity[i]: embs[i].astype(float).tolist() for i in range(len(id_to_entity))}

        # Persist
        target_dir = os.path.join(output_dir, 'pykeen_global')
        os.makedirs(target_dir, exist_ok=True)
        # Entity embeddings JSON saving removed per user request
        try:
            result.save_to_directory(os.path.join(target_dir, 'model'))
        except Exception as e:
            logger.warning(f"Could not save PyKEEN model: {e}")

        # Attach to graph and global cache
        # for ent, vec in embeddings.items():
        #     if graph.has_node(ent):
        #         # graph.nodes[ent]['kge_embedding'] = vec
        GLOBAL_ENTITY_EMBEDDINGS = embeddings

        logger.info(f"Global KGE embeddings cached for {len(embeddings)} entities")
        return embeddings
    except Exception as e:
        logger.warning(f"Global KGE training failed: {e}")
        return {}

# =============================================================================
# END OF PYVIS VISUALIZATION & RAG EMBEDDINGS
# =============================================================================

# PyKEEN Link Discovery Configuration
PYKEEN_CONFIG = {
    'min_relations_threshold': 3,  # Much lower threshold
    'embedding_dim': 32,
    'num_epochs': 25,
    'learning_rate': 0.01,
    'early_stopping_patience': 5,
    'early_stopping_frequency': 3,
    # Removed 'early_stopping_delta' - this parameter doesn't exist in PyKEEN
    'max_candidates': 100,  # More candidates
    'high_similarity_threshold': 0.8,   # more conservative
    'medium_similarity_threshold': 0.6,
    'low_similarity_threshold': 0.4,
    'min_context_overlap': 1,  # Lower requirement
    'base_decision_threshold': 0.4,     # Raised since we're more precise now
    'context_decision_threshold': 0.3,  # Raised
    'string_decision_threshold': 0.25,  # Raised
    'string_similarity_weight': 0.5,
    'embedding_similarity_weight': 0.3,
    'context_overlap_weight': 0.2,
    'max_context_contribution': 0.3,
    'pykeen_similarity_boost': 0.8,
    'coreference_threshold': 0.85,      # stricter merging
    'substring_similarity_boost': 0.9,
    'acronym_similarity_boost': 0.95,
    'title_name_similarity_boost': 0.8,
    'partial_name_similarity_boost': 0.75,
    'organization_similarity_boost': 0.7,
    # Speech-level cleanup: entities with <min_mentions must appear in a relation to be kept
    'min_mentions_per_entity': 2,
    'string_merge_min': 0.85            # minimum base string similarity for merges (unless acronym)
}

# Global cache for entity embeddings trained once across the full graph
GLOBAL_ENTITY_EMBEDDINGS: Dict[str, List[float]] = {}

def _cosine_similarity_safe(vec1: List[float], vec2: List[float]) -> Optional[float]:
    """Compute cosine similarity with basic safety checks."""
    try:
        import numpy as _np
        a = _np.asarray(vec1, dtype=float)
        b = _np.asarray(vec2, dtype=float)
        if a.shape != b.shape or a.size == 0:
            return None
        denom = (a @ a) ** 0.5 * (b @ b) ** 0.5
        if denom == 0:
            return None
        return float((a @ b) / denom)
    except Exception:
        return None

# Use shared configuration for directories
INPUT_DIR = _DEFAULT_CONFIG.get('input_dir')
OUTPUT_DIR = _DEFAULT_CONFIG.get('output_dir')

# Abstract concepts for WordNet classification
ABSTRACT_ROOT_SYNSETS = {
    'abstraction.n.06', 'psychological_feature.n.01', 'state.n.02', 'event.n.01',
    'phenomenon.n.01', 'attribute.n.02', 'group.n.01', 'cognition.n.01',
    'communication.n.01', 'relation.n.01', 'measure.n.02', 'act.n.02',
    'process.n.06', 'concept.n.01',
}

# Initialize LLM and transformer for extraction (matching original exactly)
if LLM_AVAILABLE:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=float(os.getenv("LLM_TEMPERATURE", 0)),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", 4000)),
        top_p=float(os.getenv("LLM_TOP_P", 0.9)),
        presence_penalty=float(os.getenv("LLM_PRESENCE_PENALTY", 0.3)),
        frequency_penalty=float(os.getenv("LLM_FREQUENCY_PENALTY", 0.3))
    )
else:
    llm = None

@dataclass
class ChunkExtractionTask:
    """Task for processing chunk entity/relation extraction"""
    chunk_id: str
    chunk_text: str
    entities: List[str]
    abstract_concepts: List[str]

@dataclass
class AgentDependencies:
    """Shared dependencies for all agents"""
    graph: nx.DiGraph
    nlp_model: Any
    speakers_seen: set
    total_chunks: int = 0
    total_speeches: int = 0
    extraction_tasks: List[ChunkExtractionTask] = None

    def __post_init__(self):
        if self.extraction_tasks is None:
            self.extraction_tasks = []

@dataclass
class SemanticKGResults:
    """Results from semantic knowledge graph generation with communities"""
    base_graph: nx.DiGraph
    entity_graph: nx.Graph
    communities: Dict[str, int]
    subcommunities: Dict[str, int]
    enhanced_graph: nx.DiGraph
    statistics: Dict[str, Any]
    processing_time: float
    community_quality: CommunityQualityMetrics

# =============================================================================
# TOPIC SUMMARIZATION FUNCTIONS
# =============================================================================

# Removed unused GroqModel instance (PydanticAI not used)

# Main orchestrator agent - simplified without tools
# Removed unused orchestrator_agent (PydanticAI agent not used)

# Helper functions for topic summarization
def truncate_text_for_llm(text: str, max_chars: int = 15000) -> str:
    """Truncate text to fit LLM context window with graceful degradation"""
    if len(text) <= max_chars:
        return text
    
    # Try to break at sentence boundaries
    sentences = text.split('. ')
    truncated = ""
    for sentence in sentences:
        if len(truncated) + len(sentence) + 1 <= max_chars:
            truncated += sentence + ". "
        else:
            break
    
    # If we couldn't fit any complete sentences, just truncate
    if not truncated:
        truncated = text[:max_chars]
    
    return truncated.strip()

def clean_llm_output(text: str) -> str:
    """Clean and format LLM output"""
    if not text:
        return ""
    
    text = text.strip()
    
    # Remove common prefixes that LLMs sometimes add
    prefixes_to_remove = [
        "Title:", "title:", "TITLE:",
        "Summary:", "summary:", "SUMMARY:",
        "Topic:", "topic:", "TOPIC:",
        "The title is:", "The summary is:",
        "Here is the title:", "Here is the summary:",
        "Here's the title:", "Here's the summary:",
        # More verbose meta-introductions we don't want in summaries
        "Here is a concise summary of", "Here is a brief summary of",
        "Here is a short summary of", "Here is a summary of",
        "Here's a concise summary of", "Here's a brief summary of",
        "Here's a short summary of", "Here's a summary of",
        "This is a concise summary of", "This is a brief summary of",
        "This is a short summary of", "This is a summary of",
        "Below is a concise summary of", "Below is a brief summary of",
        "Below is a short summary of", "Below is a summary of",
        "The following is a concise summary of",
        "The following is a brief summary of",
        "The following is a short summary of",
        "The following is a summary of"
    ]
    
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # Remove quotes if the entire text is wrapped in them
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    
    # Remove trailing periods from titles (but not summaries)
    # This is a simple heuristic - titles are usually shorter
    if len(text) < 100 and text.endswith('.'):
        text = text[:-1]
    
    return text.strip()

async def generate_title_internal(llm: ChatGroq, text: str) -> str:
    """Generate title for given text using LLM"""
    
    truncated_text = truncate_text_for_llm(text, max_chars=12000)
    
    title_prompt = ChatPromptTemplate.from_template(
        """Please generate a concise, descriptive title (maximum 10 words) for the following content about political/policy discussions. 
        The title should capture the main topic or theme without being too generic.
        
        Content:
        {text}
        
        Title:"""
    )
    
    try:
        # Create the chain
        chain = title_prompt | llm
        
        # Get the title
        response = await chain.ainvoke({"text": truncated_text})
        
        if hasattr(response, 'content'):
            title = response.content
        else:
            title = str(response)
        
        # Clean the output
        title = clean_llm_output(title)
        
        # Ensure it's not too long
        if len(title) > 100:
            title = title[:97] + "..."
        
        return title if title else "Untitled Topic"
        
    except Exception as e:
        logger.error(f"Title generation failed: {e}")
        return "Untitled Topic"

async def summarize_text_internal(llm: ChatGroq, text: str) -> str:
    """Generate summary for given text using LLM"""
    
    truncated_text = truncate_text_for_llm(text, max_chars=12000)
    
    summary_prompt = ChatPromptTemplate.from_template(
        """Please provide a comprehensive summary (3-5 sentences) of the following political/policy content. 
        Focus on the main points, key decisions, and important themes discussed.
        
        Content:
        {text}
        
        Summary:"""
    )
    
    try:
        # Create the chain
        chain = summary_prompt | llm
        
        # Get the summary
        response = await chain.ainvoke({"text": truncated_text})
        
        if hasattr(response, 'content'):
            summary = response.content
        else:
            summary = str(response)
        
        # Clean the output
        summary = clean_llm_output(summary)
        
        return summary if summary else "No summary available."
        
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return "No summary available."

# Additional helper functions for topic summarization
async def find_entities_for_community_async(graph: nx.DiGraph, community_node_id: str) -> List[str]:
    """Find entity IDs that belong to a community"""
    entity_ids = []
    
    # Look for entity nodes connected to this community node
    if graph.has_node(community_node_id):
        for neighbor in graph.neighbors(community_node_id):
            node_data = graph.nodes[neighbor]
            if node_data.get('node_type') == 'ENTITY':
                entity_ids.append(neighbor)
    
    return entity_ids

async def find_chunks_for_entities_async(graph: nx.DiGraph, entity_ids: List[str]) -> List[str]:
    """Find chunk IDs connected to the given entity IDs"""
    chunk_ids = set()
    
    for entity_id in entity_ids:
        if graph.has_node(entity_id):
            for neighbor in graph.neighbors(entity_id):
                node_data = graph.nodes[neighbor]
                if node_data.get('node_type') == 'CHUNK':
                    chunk_ids.add(neighbor)
    
    return list(chunk_ids)

async def sort_chunks_by_global_order_async(graph: nx.DiGraph, chunk_ids: List[str]) -> List[str]:
    """Sort chunks by their global order (speech_order, chunk_order)"""
    
    chunk_data = []
    for chunk_id in chunk_ids:
        if graph.has_node(chunk_id):
            node_data = graph.nodes[chunk_id]
            speech_order = node_data.get('speech_order', 0)
            chunk_order = node_data.get('chunk_order', 0)
            chunk_data.append((chunk_id, speech_order, chunk_order))
    
    # Sort by speech_order first, then chunk_order
    chunk_data.sort(key=lambda x: (x[1], x[2]))
    
    return [chunk_id for chunk_id, _, _ in chunk_data]

async def concatenate_chunk_texts_async(graph: nx.DiGraph, chunk_ids: List[str]) -> str:
    """Concatenate text from multiple chunks in order"""
    
    texts = []
    for chunk_id in chunk_ids:
        if graph.has_node(chunk_id):
            node_data = graph.nodes[chunk_id]
            chunk_text = node_data.get('text', '')
            if chunk_text:
                texts.append(chunk_text)
    
    return ' '.join(texts)

async def collect_community_tasks_async(graph: nx.DiGraph) -> List[SummarizationTask]:
    """Collect summarization tasks for all community nodes"""
    
    tasks = []
    community_nodes = []
    
    # Find all community nodes (COMMUNITY_X)
    for node_id, node_data in graph.nodes(data=True):
        if (node_data.get('node_type') == 'COMMUNITY' and 
            isinstance(node_id, str) and 
            node_id.startswith('COMMUNITY_')):
            community_nodes.append((node_id, node_data))
    
    logger.info(f"Found {len(community_nodes)} community nodes for summarization")
    
    for community_node_id, community_data in community_nodes:
        try:
            # Extract community ID from node name (e.g., "COMMUNITY_0" -> 0)
            community_id = int(community_node_id.split('_')[1])
            
            # Find entities for this community
            entity_ids = await find_entities_for_community_async(graph, community_node_id)
            
            if not entity_ids:
                logger.warning(f"No entities found for community {community_id}")
                continue
            
            # Find chunks for entities
            chunk_ids = await find_chunks_for_entities_async(graph, entity_ids)
            
            if not chunk_ids:
                logger.warning(f"No chunks found for community {community_id}")
                continue
            
            # Sort chunks by global order
            sorted_chunk_ids = await sort_chunks_by_global_order_async(graph, chunk_ids)
            
            # Concatenate chunk texts
            concatenated_text = await concatenate_chunk_texts_async(graph, sorted_chunk_ids)
            
            if not concatenated_text.strip():
                logger.warning(f"No text found for community {community_id}")
                continue
            
            # Create summarization task
            task = SummarizationTask(
                task_id=f"community_{community_id}",
                community_id=community_id,
                subcommunity_id=None,
                is_topic=True,
                concatenated_text=concatenated_text,
                chunk_ids=sorted_chunk_ids,
                entity_ids=entity_ids
            )
            
            tasks.append(task)
            logger.info(f"Created task for community {community_id}: {len(entity_ids)} entities, {len(sorted_chunk_ids)} chunks, {len(concatenated_text)} chars")
            
        except Exception as e:
            logger.error(f"Error creating task for community {community_node_id}: {e}")
            continue
    
    return tasks

async def collect_subcommunity_tasks_async(graph: nx.DiGraph) -> List[SummarizationTask]:
    """Collect summarization tasks for all subcommunity nodes"""
    
    tasks = []
    subcommunity_nodes = []
    
    # Find all subcommunity nodes (SUBCOMMUNITY_X_Y)
    for node_id, node_data in graph.nodes(data=True):
        if (node_data.get('node_type') == 'SUBCOMMUNITY' and 
            isinstance(node_id, str) and 
            node_id.startswith('SUBCOMMUNITY_')):
            subcommunity_nodes.append((node_id, node_data))
    
    logger.info(f"Found {len(subcommunity_nodes)} subcommunity nodes for summarization")
    
    for subcommunity_node_id, subcommunity_data in subcommunity_nodes:
        try:
            # Extract IDs from node name (e.g., "SUBCOMMUNITY_0_1" -> community=0, subcommunity=1)
            parts = subcommunity_node_id.split('_')
            if len(parts) >= 3:
                community_id = int(parts[1])
                subcommunity_id = int(parts[2])
            else:
                logger.warning(f"Invalid subcommunity node name format: {subcommunity_node_id}")
                continue
            
            # Find entities for this subcommunity
            entity_ids = await find_entities_for_community_async(graph, subcommunity_node_id)
            
            if not entity_ids:
                logger.warning(f"No entities found for subcommunity {community_id}_{subcommunity_id}")
                continue
            
            # Find chunks for entities
            chunk_ids = await find_chunks_for_entities_async(graph, entity_ids)
            
            if not chunk_ids:
                logger.warning(f"No chunks found for subcommunity {community_id}_{subcommunity_id}")
                continue
            
            # Sort chunks by global order
            sorted_chunk_ids = await sort_chunks_by_global_order_async(graph, chunk_ids)
            
            # Concatenate chunk texts
            concatenated_text = await concatenate_chunk_texts_async(graph, sorted_chunk_ids)
            
            if not concatenated_text.strip():
                logger.warning(f"No text found for subcommunity {community_id}_{subcommunity_id}")
                continue
            
            # Create summarization task
            task = SummarizationTask(
                task_id=f"subcommunity_{community_id}_{subcommunity_id}",
                community_id=community_id,
                subcommunity_id=subcommunity_id,
                is_topic=False,
                concatenated_text=concatenated_text,
                chunk_ids=sorted_chunk_ids,
                entity_ids=entity_ids
            )
            
            tasks.append(task)
            logger.info(f"Created task for subcommunity {community_id}_{subcommunity_id}: {len(entity_ids)} entities, {len(sorted_chunk_ids)} chunks, {len(concatenated_text)} chars")
            
        except Exception as e:
            logger.error(f"Error creating task for subcommunity {subcommunity_node_id}: {e}")
            continue
    
    return tasks

async def generate_title_and_summary_with_semaphore(llm: ChatGroq, task: SummarizationTask, semaphore: asyncio.Semaphore) -> SummarizationTask:
    """Generate title and summary for a task with semaphore control"""
    
    async with semaphore:
        try:
            logger.info(f"Processing task {task.task_id}...")
            
            # Generate title and summary concurrently
            title_task = generate_title_internal(llm, task.concatenated_text)
            summary_task = summarize_text_internal(llm, task.concatenated_text)
            
            title, summary = await asyncio.gather(title_task, summary_task)
            
            task.title = title
            task.summary = summary
            
            logger.info(f"Completed task {task.task_id}: '{title[:50]}...'")
            
        except Exception as e:
            logger.error(f"Failed to process task {task.task_id}: {e}")
            task.title = "Processing Failed"
            task.summary = f"Error during processing: {str(e)}"
    
    return task

async def process_all_summarization_tasks_internal(llm: ChatGroq, tasks: List[SummarizationTask]) -> Dict[str, Any]:
    """Process all summarization tasks in parallel with semaphore control"""
    
    if not tasks:
        return {
            "tasks_processed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "errors": ["No tasks to process"]
        }
    
    logger.info(f"Processing {len(tasks)} summarization tasks with max concurrency {MAX_CONCURRENT_SUMMARIES}")
    
    # Create semaphore for controlling concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SUMMARIES)
    
    # Process all tasks concurrently
    processed_tasks = await asyncio.gather(*[
        generate_title_and_summary_with_semaphore(llm, task, semaphore)
        for task in tasks
    ])
    
    # Count results
    completed = sum(1 for task in processed_tasks if task.title and task.title != "Processing Failed")
    failed = len(processed_tasks) - completed
    
    logger.info(f"Summarization complete: {completed}/{len(processed_tasks)} successful")
    
    return {
        "tasks_processed": len(processed_tasks),
        "tasks_completed": completed,
        "tasks_failed": failed,
        "processed_tasks": processed_tasks,
        "errors": []
    }

async def update_community_node_with_summary_async(graph: nx.DiGraph, community_node_id: str,
                                                  title: str = "", summary: str = "", chunk_ids: List[str] = None, 
                                                  entity_ids: List[str] = None) -> str:
    """Update a community node with title and summary"""
    
    if not graph.has_node(community_node_id):
        logger.warning(f"Community node {community_node_id} not found in graph")
        return community_node_id
    
    # Update node data
    node_data = graph.nodes[community_node_id]
    if title:
        node_data['title'] = title
    if summary:
        node_data['summary'] = summary
    if chunk_ids:
        node_data['chunk_ids'] = chunk_ids
    if entity_ids:
        node_data['entity_ids'] = entity_ids
    
    # Mark as updated
    node_data['has_summary'] = True
    node_data['updated_at'] = datetime.now().isoformat()
    
    logger.info(f"Updated {community_node_id} with title: '{title[:50]}...'")
    
    return community_node_id

async def create_all_topic_nodes(graph: nx.DiGraph, processed_tasks: List[SummarizationTask]) -> Dict[str, Any]:
    """Create topic and subtopic nodes from processed tasks"""
    
    topics_updated = 0
    subtopics_updated = 0
    
    for task in processed_tasks:
        try:
            if task.is_topic:
                # Update community node
                community_node_id = f"COMMUNITY_{task.community_id}"
                await update_community_node_with_summary_async(
                    graph, community_node_id, task.title, task.summary, 
                    task.chunk_ids, task.entity_ids
                )
                topics_updated += 1
            else:
                # Update subcommunity node
                subcommunity_node_id = f"SUBCOMMUNITY_{task.community_id}_{task.subcommunity_id}"
                await update_community_node_with_summary_async(
                    graph, subcommunity_node_id, task.title, task.summary, 
                    task.chunk_ids, task.entity_ids
                )
                subtopics_updated += 1
                
        except Exception as e:
            logger.error(f"Failed to update node for task {task.task_id}: {e}")
    
    logger.info(f"Updated {topics_updated} topics and {subtopics_updated} subtopics")
    
    return {
        "topics_updated": topics_updated,
        "subtopics_updated": subtopics_updated
    }

async def get_all_topic_nodes_async(graph: nx.DiGraph) -> List[Tuple[str, Dict[str, Any]]]:
    """Get all topic and subtopic nodes that have a title or summary (for similarity/embedding)."""
    topic_nodes = []
    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get('node_type')
        if node_type not in ('COMMUNITY', 'SUBCOMMUNITY', 'TOPIC', 'SUBTOPIC'):
            continue
        if node_data.get('title') or node_data.get('summary') or node_data.get('community_summary'):
            topic_nodes.append((node_id, node_data))
    return topic_nodes

def _topic_text_for_similarity(node_data: Dict[str, Any]) -> str:
    """Text used for topic similarity: prefer actual summary over title."""
    s = (node_data.get('summary') or node_data.get('community_summary') or '').strip()
    if s:
        return s
    return (node_data.get('title') or node_data.get('name') or '').strip()


async def find_similar_topic_pairs_async(topic_nodes: List[Tuple[str, Dict[str, Any]]]) -> List[SimilarTopicPair]:
    """Find similar topic pairs using summary-based embedding similarity when available, else Levenshtein on title."""
    similar_pairs: List[SimilarTopicPair] = []
    n = len(topic_nodes)

    # Prefer embedding-based similarity on actual summary (or title fallback) when available
    if n >= 2 and EMBEDDINGS_AVAILABLE:
        texts = []
        for _id, data in topic_nodes:
            t = _topic_text_for_similarity(data)
            texts.append(t if t else '')
        if sum(1 for t in texts if t) >= 2:
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                from sklearn.metrics.pairwise import cosine_similarity
                emb = model.encode(texts)
                sim_matrix = cosine_similarity(emb)
                for i in range(n):
                    for j in range(i + 1, n):
                        if not texts[i] or not texts[j]:
                            continue
                        similarity_score = float(sim_matrix[i, j])
                        if similarity_score <= 0.7:
                            continue
                        node1_id, node1_data = topic_nodes[i]
                        node2_id, node2_data = topic_nodes[j]
                        level1 = "TOPIC" if node1_data.get('node_type') in ('COMMUNITY', 'TOPIC') else "SUBTOPIC"
                        level2 = "TOPIC" if node2_data.get('node_type') in ('COMMUNITY', 'TOPIC') else "SUBTOPIC"
                        pair = SimilarTopicPair(
                            topic1_id=node1_id,
                            topic1_title=node1_data.get('title') or node1_id,
                            topic1_level=level1,
                            topic2_id=node2_id,
                            topic2_title=node2_data.get('title') or node2_id,
                            topic2_level=level2,
                            similarity_score=similarity_score,
                            levenshtein_distance=-1,
                            is_potential_duplicate=similarity_score > 0.9
                        )
                        similar_pairs.append(pair)
                similar_pairs.sort(key=lambda x: x.similarity_score, reverse=True)
                return similar_pairs
            except Exception as e:
                logger.warning("Summary embedding similarity failed, falling back to title Levenshtein: %s", e)

    # Fallback: Levenshtein on titles only
    for i, (node1_id, node1_data) in enumerate(topic_nodes):
        for j, (node2_id, node2_data) in enumerate(topic_nodes):
            if i >= j:
                continue
            title1 = node1_data.get('title', '') or ''
            title2 = node2_data.get('title', '') or ''
            if not title1 or not title2:
                continue
            distance = _levenshtein_distance_safe(title1.lower(), title2.lower())
            max_len = max(len(title1), len(title2))
            similarity_score = (1 - (distance / max_len)) if max_len > 0 else 1.0
            if similarity_score <= 0.7:
                continue
            level1 = "TOPIC" if node1_data.get('node_type') in ('COMMUNITY', 'TOPIC') else "SUBTOPIC"
            level2 = "TOPIC" if node2_data.get('node_type') in ('COMMUNITY', 'TOPIC') else "SUBTOPIC"
            similar_pairs.append(SimilarTopicPair(
                topic1_id=node1_id,
                topic1_title=title1,
                topic1_level=level1,
                topic2_id=node2_id,
                topic2_title=title2,
                topic2_level=level2,
                similarity_score=similarity_score,
                levenshtein_distance=distance,
                is_potential_duplicate=similarity_score > 0.9
            ))
    similar_pairs.sort(key=lambda x: x.similarity_score, reverse=True)
    return similar_pairs

async def check_topic_title_similarity_simple(graph: nx.DiGraph) -> Dict[str, Any]:
    """Check for similar topics using summary-based embedding similarity when available, else Levenshtein on titles."""
    logger.info("Checking for similar topics (summary-based embedding or title Levenshtein)...")
    
    # Get all topic nodes
    topic_nodes = await get_all_topic_nodes_async(graph)
    
    if len(topic_nodes) < 2:
        logger.info("Not enough topics to compare")
        return {
            "similar_pairs_found": 0,
            "potential_duplicates_found": 0,
            "similar_pairs": []
        }
    
    # Find similar pairs
    similar_pairs = await find_similar_topic_pairs_async(topic_nodes)
    
    # Count potential duplicates
    potential_duplicates = sum(1 for pair in similar_pairs if pair.is_potential_duplicate)
    
    logger.info(f"Found {len(similar_pairs)} similar pairs, {potential_duplicates} potential duplicates")
    
    return {
        "similar_pairs_found": len(similar_pairs),
        "potential_duplicates_found": potential_duplicates,
        "similar_pairs": similar_pairs
    }

# Core topic summarization function
async def run_topic_summarization_on_graph(graph: nx.DiGraph, graph_file_path: str, output_file_path: str = None) -> Dict[str, Any]:
    """
    Run topic summarization on an existing semantic knowledge graph.
    
    Args:
        graph: The NetworkX DiGraph with communities
        graph_file_path: Path to the original graph file (for reference)
        output_file_path: Path to save the updated graph (defaults to same directory)
        
    Returns:
        Processing results
    """
    logger.info("Starting Topic Summarization on Semantic Knowledge Graph")
    
    start_time = datetime.now()
    
    # Initialize ChatGroq LLM with the same configuration
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=float(os.getenv("LLM_TEMPERATURE", 0)),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", 4000)),
            top_p=float(os.getenv("LLM_TOP_P", 0.9)),
            presence_penalty=float(os.getenv("LLM_PRESENCE_PENALTY", 0.3)),
            frequency_penalty=float(os.getenv("LLM_FREQUENCY_PENALTY", 0.3))
        )
        logger.info("LLM initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise
    
    # Determine output path
    if output_file_path is None:
        output_file_path = graph_file_path.replace('.json', '_with_topics.json')
    
    try:
        # Phase 1: Collect summarization tasks for communities and subcommunities
        logger.info("Phase 1: Collecting summarization tasks...")
        community_tasks = await collect_community_tasks_async(graph)
        subcommunity_tasks = await collect_subcommunity_tasks_async(graph)
        
        all_tasks = community_tasks + subcommunity_tasks
        logger.info(f"Created {len(all_tasks)} summarization tasks ({len(community_tasks)} communities, {len(subcommunity_tasks)} subcommunities)")
        
        if not all_tasks:
            logger.warning("No summarization tasks created. Graph may not have proper community structure.")
            return {
                "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                "total_communities": 0,
                "total_subcommunities": 0,
                "topics_updated": 0,
                "subtopics_updated": 0,
                "total_chunks_processed": 0,
                "similar_pairs_found": 0,
                "potential_duplicates_found": 0,
                "similarity_file_path": None,
                "errors": ["No summarization tasks could be created"]
            }
        
        # Phase 2: Process summarization tasks in parallel
        logger.info(f"Phase 2: Processing {len(all_tasks)} summarization tasks...")
        summary_result = await process_all_summarization_tasks_internal(llm, all_tasks)
        
        # Phase 3: Update community nodes with summaries from completed tasks
        logger.info("Phase 3: Updating community nodes with summaries...")
        creation_result = await create_all_topic_nodes(graph, summary_result["processed_tasks"])
        
        # Phase 4: Check for similar topic titles using Levenshtein Distance
        logger.info("Phase 4: Checking for similar topic titles...")
        similarity_result = await check_topic_title_similarity_simple(graph)
        
        # Phase 5: Save the updated graph
        logger.info("Phase 5: Saving updated graph...")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert graph to JSON format
        # Define embedding-related attribute names to filter out
        embedding_attributes = {
            'kge_embedding', 'embedding', 'embeddings', 'vector', 'vectors',
            'embedding_vector', 'node_embedding', 'entity_embedding',
            'semantic_embedding', 'graph_embedding', 'feature_vector'
        }
        
        def filter_node_data(node_data):
            """Filter out embedding attributes from node data"""
            filtered_data = {}
            for key, value in node_data.items():
                if key.lower() not in embedding_attributes and not key.lower().endswith('_embedding'):
                    filtered_data[key] = value
            return filtered_data
        
        graph_data = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [
                {
                    "id": str(node_id),
                    "data": filter_node_data(node_data)
                }
                for node_id, node_data in graph.nodes(data=True)
            ],
            "links": [
                {
                    "source": str(source),
                    "target": str(target),
                    "data": edge_data
                }
                for source, target, edge_data in graph.edges(data=True)
            ]
        }
        
        # Save updated graph
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False, default=json_serialize_dates)
        
        # Calculate total chunks processed
        total_chunks = sum(len(task.chunk_ids) for task in summary_result["processed_tasks"])
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        final_results = {
            "processing_time_seconds": processing_time,
            "total_communities": len(community_tasks),
            "total_subcommunities": len(subcommunity_tasks), 
            "topics_updated": creation_result["topics_updated"],
            "subtopics_updated": creation_result["subtopics_updated"],
            "total_chunks_processed": total_chunks,
            "similar_pairs_found": similarity_result["similar_pairs_found"],
            "potential_duplicates_found": similarity_result["potential_duplicates_found"],
            "similarity_file_path": output_file_path.replace('.json', '_similarity_analysis.json'),
            "output_file_path": output_file_path,
            "errors": []
        }
        
        # Save similarity analysis if there are similar pairs
        if similarity_result["similar_pairs"]:
            similarity_data = {
                "analysis_timestamp": datetime.now().isoformat(),
                "total_pairs_analyzed": len(similarity_result["similar_pairs"]),
                "similar_pairs_found": similarity_result["similar_pairs_found"],
                "potential_duplicates_found": similarity_result["potential_duplicates_found"],
                "similarity_threshold": 0.7,
                "duplicate_threshold": 0.9,
                "similar_pairs": [
                    {
                        "topic1_id": pair.topic1_id,
                        "topic1_title": pair.topic1_title,
                        "topic1_level": pair.topic1_level,
                        "topic2_id": pair.topic2_id,
                        "topic2_title": pair.topic2_title,
                        "topic2_level": pair.topic2_level,
                        "similarity_score": pair.similarity_score,
                        "levenshtein_distance": pair.levenshtein_distance,
                        "is_potential_duplicate": pair.is_potential_duplicate
                    }
                    for pair in similarity_result["similar_pairs"]
                ]
            }
            
            with open(final_results["similarity_file_path"], 'w', encoding='utf-8') as f:
                json.dump(similarity_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Topic summarization completed in {processing_time:.2f} seconds")
        logger.info(f"Updated {creation_result['topics_updated']} topics and {creation_result['subtopics_updated']} subtopics")
        logger.info(f"Output saved to: {output_file_path}")
        
        return final_results
        
    except Exception as e:
        logger.error(f"Topic summarization failed: {e}")
        return {
            "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
            "total_communities": 0,
            "total_subcommunities": 0,
            "topics_updated": 0,
            "subtopics_updated": 0,
            "total_chunks_processed": 0,
            "similar_pairs_found": 0,
            "potential_duplicates_found": 0,
            "similarity_file_path": None,
            "errors": [str(e)]
        }

# Utility functions for concept and entity extraction
def get_wordnet_pos(treebank_tag):
    """Helper to convert Penn Treebank POS tags to WordNet POS tags"""
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN

def is_abstract_noun(word_lemma):
    """Checks if a noun lemma might be abstract by checking its WordNet hypernyms."""
    if not word_lemma:
        return False

    synsets = wn.synsets(word_lemma, pos=wn.NOUN)
    if not synsets:
        return False

    for synset in synsets:
        if synset.name() in ABSTRACT_ROOT_SYNSETS:
            return True
        for hypernym_path in synset.hypernym_paths():
            for hypernym in hypernym_path:
                if hypernym.name() in ABSTRACT_ROOT_SYNSETS:
                    return True
    return False

async def build_knowledge_graph(deps: AgentDependencies, input_dir: str = None) -> Dict[str, Any]:
    """Main orchestrator: builds lexical graph then extracts entities/relations in parallel"""
    
    if input_dir is None:
        input_dir = INPUT_DIR
    
    # Phase 1: Sequential lexical graph construction
    logger.info("Phase 1: Building lexical graph structure...")
    lexical_result = await build_lexical_graph(deps, input_dir)
    
    # Generate claims for chunks before entity/relation extraction (async)
    try:
        claims_stats = await generate_claims_for_chunks_async(deps.graph)
        logger.info(f"Claims generated for chunks (valid={claims_stats.get('valid', 0)}, invalid={claims_stats.get('invalid', 0)})")
    except Exception as e:
        logger.warning(f"Chunk claims generation failed before extraction: {e}")
    
    # Phase 2: Parallel entity/relation extraction
    logger.info(f"Phase 2: Processing {len(deps.extraction_tasks)} chunks for entity/relation extraction...")
    extraction_result = await extract_all_entities_relations(deps)
    
    return {
        "lexical_result": lexical_result,
        "extraction_result": extraction_result,
        "total_extraction_tasks": len(deps.extraction_tasks)
    }

async def build_lexical_graph(deps: AgentDependencies, input_dir: str) -> Dict[str, Any]:
    """Phase 1: Build the complete lexical graph structure sequentially"""
    results = {"documents_processed": 0, "total_speeches": 0, "total_chunks": 0, "errors": []}
    
    try:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory {input_dir} not found")
            
        filenames = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        logger.info(f"Found {len(filenames)} files to process")
        
        # Process documents sequentially to build lexical structure
        for filename in filenames:
            if deps.total_speeches >= SPEECH_LIMIT and USE_SPEECH_LIMIT:
                break
                
            doc_result = await process_single_document_lexical(deps, filename, input_dir)
            results["documents_processed"] += 1
            results["total_speeches"] += doc_result.get("speeches_added", 0)
            results["total_chunks"] += doc_result.get("chunks_added", 0)
            
            if doc_result.get("errors"):
                results["errors"].extend(doc_result["errors"])
                
        return results
        
    except Exception as e:
        error_msg = f"Error in build_lexical_graph: {str(e)}"
        logger.error(error_msg, exc_info=True)
        results["errors"].append(error_msg)
        return results

async def process_single_document_lexical(deps: AgentDependencies, filename: str, input_dir: str) -> Dict[str, Any]:
    """Process a single document and build lexical structure only"""
    doc_metadata = re.search(r'CRE-(\d+)-(\d{4}-\d{2}-\d{2})-ITM-(\d{3})_EN\.txt', filename)
    
    if not doc_metadata:
        logger.warning(f"Skipping {filename} - doesn't match expected pattern")
        return {"error": f"Skipping {filename} - doesn't match expected pattern", "speeches_added": 0, "chunks_added": 0}
    
    # Create document identifier
    doc_id = f"DOC_{doc_metadata.group(1)}_{doc_metadata.group(2)}_{doc_metadata.group(3)}"
    doc_date = doc_metadata.group(2)
    doc_date_obj = datetime.strptime(doc_date, "%Y-%m-%d").date()
    
    try:
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            
        logger.info(f"Processing document {doc_id} with {len(text)} characters")
        
        # Add document as a node
        deps.graph.add_node(doc_id, 
                           node_type="DOCUMENT", 
                           graph_type="lexical_graph",
                           filename=filename,
                           date=doc_date,
                           text_length=len(text))
        
        # Extract speeches and build chunks (lexical structure only)
        speeches_result = await extract_speeches_lexical(deps, text, doc_id, doc_date, doc_date_obj)
        
        return {
            "document_id": doc_id,
            "speeches_added": speeches_result.get("speeches_count", 0),
            "chunks_added": speeches_result.get("chunks_count", 0),
            "errors": speeches_result.get("errors", [])
        }
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)
        return {"error": f"Error processing {filename}: {str(e)}", "speeches_added": 0, "chunks_added": 0}

async def extract_speeches_lexical(deps: AgentDependencies, text: str, doc_id: str, doc_date: str, doc_date_obj) -> Dict[str, Any]:
    """Extract speeches and create chunks (lexical structure only, no LLM calls)"""
    speech_count = 0
    chunk_count = 0
    
    # Extract speeches with pattern: [EN] FirstName | LastName said from the PARTY that...
    for line_num, speech in enumerate(text.split('\n')):
        if not speech.strip():
            continue
            
        speaker_match = re.search(r'\[EN\]\s*([^|]+) \| ([^|]+) said from the (.*?) that (.+)', speech)
        
        if speaker_match and (not USE_SPEECH_LIMIT or deps.total_speeches < SPEECH_LIMIT):
            first_name = speaker_match.group(1).strip()
            last_name = speaker_match.group(2).strip()
            party = speaker_match.group(3).strip()
            speech_content = speaker_match.group(4).strip()
            
            speaker_name = f"{first_name} {last_name}"
            speaker_id = f"SPEAKER_{speaker_name.replace(' ', '_')}"
            speech_id = f"SPEECH_{doc_id}_{line_num}_{speaker_name.replace(' ', '_')}"
            
            # Add speaker if not already added
            if speaker_id not in deps.speakers_seen:
                deps.graph.add_node(speaker_id,
                                  node_type="SPEAKER",
                                  graph_type="lexical_graph",
                                  speaker_name=speaker_name,
                                  first_name=first_name,
                                  last_name=last_name,
                                  party=party)
                deps.speakers_seen.add(speaker_id)
                logger.info(f"  Added new speaker: {speaker_name} ({party})")
            
            # Add speech as a node
            deps.graph.add_node(speech_id,
                              node_type="SPEECH",
                              graph_type="lexical_graph",
                              content=speech_content,
                              content_length=len(speech_content),
                              line_number=line_num,
                              document_date=doc_date,
                              date=doc_date_obj,
                              local_speech_order=speech_count,
                              global_speech_order=deps.total_speeches)
            
            # Add edges
            deps.graph.add_edge(doc_id, speech_id, label="HAS_SPEECH", graph_type="lexical_graph")
            deps.graph.add_edge(speaker_id, speech_id, label="HAS_SPEECH", graph_type="lexical_graph")
            
            logger.info(f"  Added speech: {speaker_name} ({party}) -> {speech_content[:50]}...")
            
            # Process speech into chunks (structure only, no extraction)
            chunks_added = await create_speech_chunks_lexical(deps, speech_content, speech_id)
            chunk_count += chunks_added
            
            speech_count += 1
            deps.total_speeches += 1
            
            if USE_SPEECH_LIMIT and deps.total_speeches >= SPEECH_LIMIT:
                break
    
    return {"speeches_count": speech_count, "chunks_count": chunk_count, "errors": []}

def extract_concepts_and_entities(sentences):
    """
    Extract and deduplicate concepts and entities from sentences.
    
    Args:
        sentences (list): List of sentence strings
        
    Returns:
        tuple: (abstract_concepts, entities) - both as deduplicated lists
    """
    # Combine all sentences into one text
    text = " ".join(sentences)
    
    # Extract abstract concepts and entities
    abstract_concepts = extract_abstract_concepts_sync(text)
    entities = extract_entities_sync(text)
    
    # Deduplicate while preserving order
    abstract_concepts = list(dict.fromkeys(abstract_concepts))
    entities = list(dict.fromkeys(entities))
    
    return abstract_concepts, entities

async def create_speech_chunks_lexical(deps: AgentDependencies, speech_content: str, speech_id: str) -> int:
    """Create chunks for a speech using EXACT same approach as original kg.py"""
    
    # Sentence tokenization (EXACT match to original)
    doc = deps.nlp_model(speech_content)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    logger.info(f"    Speech has {len(sentences)} sentences")
    
    chunks_added = 0
    
    # Create larger, non-overlapping chunks to reduce community fragmentation
    chunk_size = 8  # Larger chunks for better topic coherence
    if len(sentences) >= chunk_size:
        # Non-overlapping chunks to reduce duplication
        chunks = [sentences[i:i+chunk_size] for i in range(0, len(sentences), chunk_size)]
        # Include any remaining sentences in the last chunk
        if len(sentences) % chunk_size != 0 and len(chunks) > 0:
            chunks[-1].extend(sentences[len(chunks) * chunk_size:])
        logger.info(f"    Creating {len(chunks)} chunks from {len(sentences)} sentences")
        
        for chunk_num, chunk in enumerate(chunks):
            chunk_id = f"CHUNK_{speech_id}_{chunk_num}"
            
            # Extract abstract concepts and entities from chunk (EXACT match to original)
            abstract_concepts, entities = extract_concepts_and_entities(chunk)
            
            # Add chunk node (structure only, relations will be added in Phase 2)
            # Add chronological ordering for pipeline efficiency
            global_chunk_order = deps.total_speeches * 1000 + chunk_num  # Ensures global chronological order
            
            deps.graph.add_node(chunk_id, 
                              node_type="CHUNK", 
                              graph_type="lexical_graph",
                              sentences=chunk,
                              sentence_count=len(chunk),
                              chunk_number=chunk_num,
                              speech_order=deps.total_speeches,  # Global speech order
                              global_chunk_order=global_chunk_order,  # Global chronological order
                              initial_abstract_concepts=abstract_concepts,
                              initial_entities=entities)
            
            # Connect chunk to speech
            deps.graph.add_edge(speech_id, chunk_id, label="HAS_CHUNK", graph_type="lexical_graph")
            
            # Add to extraction tasks for Phase 2
            chunk_text = " ".join(chunk)
            deps.extraction_tasks.append(ChunkExtractionTask(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                entities=entities,
                abstract_concepts=abstract_concepts
            ))
            
            chunks_added += 1
            deps.total_chunks += 1
            
    elif len(sentences) > 0:
        # For speeches with fewer than 8 sentences, create a single chunk
        chunk_id = f"CHUNK_{speech_id}_0"
        
        # Extract abstract concepts and entities from chunk (EXACT match to original)
        abstract_concepts, entities = extract_concepts_and_entities(sentences)
        
        # Add chronological ordering for pipeline efficiency
        global_chunk_order = deps.total_speeches * 1000 + 0  # Single chunk gets chunk_number 0
        
        deps.graph.add_node(chunk_id, 
                          node_type="CHUNK", 
                          graph_type="lexical_graph",
                          sentences=sentences,
                          sentence_count=len(sentences),
                          chunk_number=0,
                          speech_order=deps.total_speeches,  # Global speech order
                          global_chunk_order=global_chunk_order,  # Global chronological order
                          initial_abstract_concepts=abstract_concepts,
                          initial_entities=entities)
        
        deps.graph.add_edge(speech_id, chunk_id, label="HAS_CHUNK", graph_type="lexical_graph")
        
        chunk_text = " ".join(sentences)
        deps.extraction_tasks.append(ChunkExtractionTask(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            entities=entities,
            abstract_concepts=abstract_concepts
        ))
        
        chunks_added += 1
        deps.total_chunks += 1
        
        logger.info(f"    Added 1 chunk for short speech {speech_id} ({len(sentences)} sentences)")
    
    return chunks_added

async def extract_all_entities_relations(deps: AgentDependencies) -> Dict[str, Any]:
    """Phase 2: Process all chunks in parallel for entity/relation extraction"""
    if not deps.extraction_tasks:
        return {"processed": 0, "successful": 0, "errors": []}
    
    # Process extraction tasks in parallel with concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTIONS)
    tasks = []
    
    for task in deps.extraction_tasks:
        tasks.append(process_extraction_task(deps, task, semaphore))
    
    logger.info(f"Processing {len(tasks)} extraction tasks with max {MAX_CONCURRENT_EXTRACTIONS} concurrent...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    successful = 0
    errors = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            error_msg = f"Task {i}: {str(result)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
        elif result.get("success"):
            successful += 1
        else:
            error_msg = f"Task {i}: {result.get('error', 'Unknown error')}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # NEW: After chunk-level extraction, perform speech-level enrichment, coref, and PyKEEN once per speech
    try:
        speech_enrichment_stats = await enrich_graph_per_speech(deps)
        logger.info(f"Speech-level enrichment complete: {speech_enrichment_stats.get('speeches_processed', 0)} speeches")
    except Exception as e:
        logger.warning(f"Speech-level enrichment failed: {e}")
        speech_enrichment_stats = {"speeches_processed": 0, "errors": [str(e)]}
    
    return {
        "processed": len(results),
        "successful": successful,
        "errors": errors + speech_enrichment_stats.get("errors", [])
    }

async def process_extraction_task(deps: AgentDependencies, task: ChunkExtractionTask, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """Process a single chunk extraction task with enhanced coreference resolution"""
    async with semaphore:
        chunk_short_id = task.chunk_id.split('_')[-2] + '_' + task.chunk_id.split('_')[-1]
        logger.info(f"      🚀 Starting {chunk_short_id}")
        
        try:
            # Extract relations using LLM
            # Prefer claims text if available and marked valid; else fall back to original chunk text
            node_meta = deps.graph.nodes.get(task.chunk_id, {})
            claims_text = node_meta.get('claims') if node_meta else None
            use_claims = bool(claims_text) and bool(node_meta.get('claims_valid', True))
            source_text = claims_text if use_claims else task.chunk_text
            raw_relations = await extract_relations_with_llm_async(source_text, task.entities, task.abstract_concepts)
            
            # IMPORTANT CHANGE: Do NOT run coref/pykeen or add graph edges here. Store raw per-chunk results only.
            chunk_data = {
                'raw_extraction': {
                    'relations': raw_relations,
                    'entities': task.entities,
                    'abstract_concepts': task.abstract_concepts
                }
            }
            deps.graph.nodes[task.chunk_id].update(chunk_data)
            
            # Mark extraction success based on raw relations presence
            deps.graph.nodes[task.chunk_id]['extraction_successful'] = bool(raw_relations)
            rel_count = len(raw_relations)
            ent_count = len(set([x for tr in raw_relations for x in (tr[0], tr[2])])) if raw_relations else 0
            logger.info(f"      ✅ Completed {chunk_short_id}: stored {ent_count} entities, {rel_count} relations (chunk-level only)")
            
            return {"success": True, "chunk_id": task.chunk_id}
            
        except Exception as e:
            logger.error(f"      ❌ Failed {chunk_short_id}: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e), "chunk_id": task.chunk_id}

async def enrich_graph_per_speech(deps: AgentDependencies) -> Dict[str, Any]:
    """Aggregate chunk-level extractions per speech, then run coref/pykeen and enrich the graph once per speech."""
    graph = deps.graph
    speech_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'SPEECH']
    speeches_processed = 0
    errors: List[str] = []
    
    for speech_id in speech_nodes:
        try:
            chunk_ids = _get_chunks_for_speech(graph, speech_id)
            if not chunk_ids:
                continue
            
            # Collect per-chunk raw relations and entity mentions
            aggregated_relations: List[Tuple[str, str, str]] = []
            relation_provenance: Dict[Tuple[str, str, str], Set[str]] = {}
            chunk_entity_map: Dict[str, Set[str]] = {}
            all_entities: Set[str] = set()
            
            for cid in chunk_ids:
                node = graph.nodes.get(cid, {})
                raw_relations = (node.get('raw_extraction') or {}).get('relations') or []
                # Track entities per chunk from relations (fallback to initial_entities as needed)
                entities_in_chunk: Set[str] = set()
                for (h, r, t) in raw_relations:
                    aggregated_relations.append((h, r, t))
                    key = (h, r, t)
                    relation_provenance.setdefault(key, set()).add(cid)
                    entities_in_chunk.add(h)
                    entities_in_chunk.add(t)
                if not entities_in_chunk:
                    initial_ents = node.get('initial_entities') or []
                    entities_in_chunk.update(initial_ents)
                if entities_in_chunk:
                    chunk_entity_map[cid] = entities_in_chunk
                    all_entities.update(entities_in_chunk)
            
            if not aggregated_relations and not all_entities:
                continue

            # Speech-level entity filtering: drop low-signal entities unless referenced in relations
            try:
                mention_counts: Dict[str, int] = {}
                for ents in chunk_entity_map.values():
                    for e in ents:
                        mention_counts[e] = mention_counts.get(e, 0) + 1

                referenced_entities: Set[str] = set()
                for (h, _, t) in aggregated_relations:
                    referenced_entities.add(h)
                    referenced_entities.add(t)

                min_mentions = PYKEEN_CONFIG.get('min_mentions_per_entity', 2)
                keep_entities: Set[str] = set()
                for e in all_entities:
                    if mention_counts.get(e, 0) >= min_mentions or e in referenced_entities:
                        keep_entities.add(e)

                if keep_entities and keep_entities != all_entities:
                    # Filter per-chunk maps
                    for cid in list(chunk_entity_map.keys()):
                        kept = chunk_entity_map[cid].intersection(keep_entities)
                        if kept:
                            chunk_entity_map[cid] = kept
                        else:
                            del chunk_entity_map[cid]
                    all_entities = keep_entities
                    if not all_entities:
                        continue
            except Exception:
                # Non-fatal; proceed without filtering if anything unexpected happens
                pass
            
            # Run coreference resolution + PyKEEN at speech level
            coref_result = resolve_coreferences_with_pykeen(aggregated_relations, list(all_entities))
            cleaned_relations: List[Tuple[str, str, str]] = coref_result.get('cleaned_relations', [])
            entity_mappings: Dict[str, str] = coref_result.get('entity_mappings', {})
            
            # Write entities/edges once per speech, preserving provenance to chunks
            await add_triplets_to_graph_for_speech(
                deps=deps,
                relations=cleaned_relations,
                entity_mappings=entity_mappings,
                speech_id=speech_id,
                chunk_entity_map=chunk_entity_map,
                relation_provenance=relation_provenance
            )
            
            # Persist detailed entity resolution info on the SPEECH node
            try:
                graph.nodes[speech_id]['entity_resolution'] = {
                    'speech_id': speech_id,
                    'chunks': sorted(list(chunk_entity_map.keys())),
                    'raw_relations_count': len(aggregated_relations),
                    'cleaned_relations_count': len(cleaned_relations),
                    'predicted_links': coref_result.get('predicted_links', []),
                    'entity_mappings': coref_result.get('entity_mappings', {}),
                    'merged_entities': coref_result.get('merged_entities', []),
                    'coreference_threshold': coref_result.get('coreference_threshold', PYKEEN_CONFIG.get('coreference_threshold')),
                    'debug_log': coref_result.get('debug_log', [])
                }
                # Optional global log aggregation for corpus-level audit
                logs = graph.graph.get('entity_resolution_logs', [])
                logs.extend(coref_result.get('debug_log', []))
                graph.graph['entity_resolution_logs'] = logs
            except Exception as _:
                pass

            speeches_processed += 1
        except Exception as e:
            logger.warning(f"Speech-level enrichment failed for {speech_id}: {e}")
            errors.append(f"{speech_id}: {e}")
            continue
    
    return {"speeches_processed": speeches_processed, "errors": errors}

async def add_triplets_to_graph_for_speech(
    deps: AgentDependencies,
    relations: List[Tuple[str, str, str]],
    entity_mappings: Dict[str, str],
    speech_id: str,
    chunk_entity_map: Dict[str, Set[str]],
    relation_provenance: Dict[Tuple[str, str, str], Set[str]]
) -> None:
    """Add nodes/edges for a speech from cleaned relations, attaching HAS_ENTITY from all relevant chunks.
    Also add SPEAKER_MENTIONS from the speech's speaker to all canonical entities appearing in this speech.
    """
    graph = deps.graph
    
    # Helper to canonicalize
    def canon(name: str) -> str:
        return entity_mappings.get(name, name)
    
    # Ensure entity nodes exist and track which chunks mention which canonical entity
    canonical_entities_in_speech: Set[str] = set()
    for chunk_id, entity_names in chunk_entity_map.items():
        for raw in entity_names:
            canonical = canon(raw)
            canonical_entities_in_speech.add(canonical)
            ent_id = canonical.replace(' ', '_').upper()
            if ent_id not in graph:
                graph.add_node(
                    ent_id,
                    node_type="ENTITY_CONCEPT",
                    graph_type="entity_relation",
                    name=canonical,
                    entity_type=determine_entity_type(canonical),
                    extracted_from=[chunk_id]
                )
            else:
                extracted_from = graph.nodes[ent_id].get('extracted_from', [])
                if chunk_id not in extracted_from:
                    extracted_from.append(chunk_id)
                    graph.nodes[ent_id]['extracted_from'] = extracted_from
            # Connect chunk to canonical entity
            if not graph.has_edge(chunk_id, ent_id):
                graph.add_edge(chunk_id, ent_id, label="HAS_ENTITY", graph_type="lexical_graph")
    
    # Add entity-entity relations with provenance
    for (h, r, t) in relations:
        h_c = canon(h).replace(' ', '_').upper()
        t_c = canon(t).replace(' ', '_').upper()
        if h_c == t_c:
            continue
        if not graph.has_edge(h_c, t_c):
            prov_chunks = sorted(list(relation_provenance.get((h, r, t), set())))
            graph.add_edge(
                h_c, t_c,
                label=r,
                graph_type="entity_relation",
                relation_type=r,
                source_name=canon(h),
                target_name=canon(t),
                extracted_from=prov_chunks
            )
        else:
            existing = graph.edges[h_c, t_c]
            prov_chunks = set(existing.get('extracted_from', [])).union(relation_provenance.get((h, r, t), set()))
            graph.edges[h_c, t_c]['extracted_from'] = sorted(list(prov_chunks))
    
    # Add SPEAKER_MENTIONS for all canonical entities in this speech
    speaker_id = _get_speaker_for_speech(graph, speech_id)
    if speaker_id:
        for canonical in canonical_entities_in_speech:
            ent_id = canonical.replace(' ', '_').upper()
            if not graph.has_edge(speaker_id, ent_id):
                graph.add_edge(
                    speaker_id,
                    ent_id,
                    label="SPEAKER_MENTIONS",
                    graph_type="entity_relation",
                    relation_type="SPEAKER_MENTIONS",
                    mentioned_in_speech=speech_id
                )

def is_grammatically_valid_concept(word: str, lemma: str, pos_tag: str) -> bool:
    """Use grammatical and linguistic rules to determine if a word is a valid abstract concept"""
    
    # 1. Basic structural filters using grammar
    if len(lemma) < 3:  # Too short to be meaningful
        return False
    
    if not lemma.isalpha():  # Must be purely alphabetic
        return False
    
    # 2. POS tag analysis - ensure it's a substantial noun
    if pos_tag not in ['NN', 'NNS']:  # Only common nouns, not proper nouns
        return False
    
    # 3. Morphological analysis - filter out function words and determiners
    # Check if word has meaningful morphological structure
    if lemma.endswith(('ing', 'ed', 'er', 'est', 'ly')):  # These are typically not abstract concepts
        return False
    
    # 4. WordNet linguistic analysis
    synsets = wn.synsets(lemma, pos=wn.NOUN)
    if not synsets:
        return False
    
    # 5. Semantic depth analysis - abstract concepts have deeper semantic hierarchies
    max_depth = 0
    has_abstract_path = False
    
    for synset in synsets:
        # Check if this synset leads to abstract concepts
        paths = synset.hypernym_paths()
        for path in paths:
            if len(path) > max_depth:
                max_depth = len(path)
            
            # Check if path contains abstract root concepts
            for hypernym in path:
                if hypernym.name() in ABSTRACT_ROOT_SYNSETS:
                    has_abstract_path = True
                    break
    
    # Abstract concepts typically have deeper semantic hierarchies (depth > 4)
    # and must connect to abstract root synsets
    return has_abstract_path and max_depth > 4

def is_pronoun_or_determiner(word: str, pos_tag: str) -> bool:
    """Check if word is pronoun, determiner or function word using POS tags"""
    function_tags = ['PRP', 'PRP$', 'WP', 'WP$', 'DT', 'WDT', 'PDT', 'CD', 'MD']
    return pos_tag in function_tags

def is_temporal_or_spatial_deictic(lemma: str) -> bool:
    """Check if word is temporal/spatial deictic using WordNet semantic relations"""
    synsets = wn.synsets(lemma, pos=wn.NOUN)
    
    for synset in synsets:
        # Check if synset is related to time.n.01 or space.n.01
        for path in synset.hypernym_paths():
            for hypernym in path:
                if hypernym.name() in ['time.n.01', 'space.n.01', 'location.n.01', 'temporal_relation.n.01']:
                    return True
    return False

def is_modal_or_auxiliary_concept(lemma: str, original_pos: str) -> bool:
    """Filter out modal and auxiliary concepts using grammatical analysis"""
    # Check if the original POS tag suggests modal/auxiliary usage
    if original_pos in ['MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return True
    
    # Check if WordNet classifies it primarily as a verb concept
    verb_synsets = wn.synsets(lemma, pos=wn.VERB)
    noun_synsets = wn.synsets(lemma, pos=wn.NOUN)
    
    # If more verb senses than noun senses, likely not a good abstract concept
    if len(verb_synsets) > len(noun_synsets):
        return True
    
    return False

def determine_entity_type(entity_name: str) -> str:
    """Determine entity type using linguistic and semantic analysis"""
    
    # Clean and prepare the entity name for analysis
    words = entity_name.lower().split()
    
    if len(words) == 0:
        return "UNKNOWN"
    
    # For multi-word entities, focus on the head noun (typically the last word)
    head_word = words[-1]
    
    # Get WordNet synsets for semantic analysis
    synsets = wn.synsets(head_word, pos=wn.NOUN)
    
    if not synsets:
        # If no noun synsets, try with the full entity name
        synsets = wn.synsets(entity_name.lower().replace(' ', '_'), pos=wn.NOUN)
    
    if not synsets:
        # Fallback: use grammatical patterns for compound entities
        if any(word in entity_name.lower() for word in ['development', 'process', 'procedure']):
            return "PROCESS"
        elif any(word in entity_name.lower() for word in ['country', 'countries', 'nation', 'state']):
            return "GEOPOLITICAL_ENTITY"
        elif any(word in entity_name.lower() for word in ['policy', 'law', 'regulation', 'rule']):
            return "POLICY"
        elif any(word in entity_name.lower() for word in ['organization', 'institution', 'agency', 'body']):
            return "ORGANIZATION"
        else:
            return "CONCEPT"
    
    # Semantic analysis using WordNet hypernyms
    semantic_types = {
        'person.n.01': 'PERSON',
        'group.n.01': 'GROUP', 
        'organization.n.01': 'ORGANIZATION',
        'institution.n.01': 'ORGANIZATION',
        'location.n.01': 'LOCATION',
        'geopolitical_entity.n.01': 'GEOPOLITICAL_ENTITY',
        'country.n.02': 'GEOPOLITICAL_ENTITY',
        'political_unit.n.01': 'GEOPOLITICAL_ENTITY',
        'event.n.01': 'EVENT',
        'act.n.02': 'ACTION',
        'activity.n.01': 'ACTIVITY', 
        'process.n.06': 'PROCESS',
        'procedure.n.01': 'PROCESS',
        'policy.n.01': 'POLICY',
        'rule.n.01': 'POLICY',
        'law.n.01': 'POLICY',
        'document.n.01': 'DOCUMENT',
        'artifact.n.01': 'ARTIFACT',
        'substance.n.01': 'SUBSTANCE',
        'abstraction.n.06': 'CONCEPT',
        'psychological_feature.n.01': 'CONCEPT',
        'communication.n.01': 'COMMUNICATION',
        'measure.n.02': 'MEASURE',
        'time_period.n.01': 'TIME_PERIOD'
    }
    
    # Check each synset's hypernym paths
    for synset in synsets:
        for path in synset.hypernym_paths():
            for hypernym in path:
                if hypernym.name() in semantic_types:
                    return semantic_types[hypernym.name()]
    
    # Default fallback
    return "CONCEPT"

def compute_entity_similarity(entity1: str, entity2: str) -> float:
    """Composite similarity that prioritizes strong signals (e.g., acronyms), with conservative type checks."""
    # Normalize
    e1_norm = entity1.lower().strip()
    e2_norm = entity2.lower().strip()
    if e1_norm == e2_norm:
        return 1.0

    def _calculate_acronym_score(str1: str, str2: str) -> float:
        tokens1 = str1.split()
        tokens2 = str2.split()
        # str1 acronym of str2
        if len(tokens1) == 1 and len(tokens1[0]) > 1 and len(tokens2) > 1:
            acronym = "".join(w[0] for w in tokens2 if w and w[0].isalpha())
            if tokens1[0] == acronym:
                return 0.98
        # str2 acronym of str1
        if len(tokens2) == 1 and len(tokens2[0]) > 1 and len(tokens1) > 1:
            acronym = "".join(w[0] for w in tokens1 if w and w[0].isalpha())
            if tokens2[0] == acronym:
                return 0.98
        return 0.0

    # 1) Acronym (highest priority)
    acronym_sim = _calculate_acronym_score(e1_norm, e2_norm)
    if acronym_sim > 0.9:
        return acronym_sim

    # 2) String similarity
    string_sim = SequenceMatcher(None, e1_norm, e2_norm).ratio()

    # 3) Token Jaccard
    tokens1 = set(e1_norm.split())
    tokens2 = set(e2_norm.split())
    inter = len(tokens1.intersection(tokens2))
    uni = len(tokens1.union(tokens2))
    token_sim = (inter / uni) if uni > 0 else 0.0

    # 4) Substring containment (length-normalized)
    substring_sim = 0.0
    if e1_norm in e2_norm or e2_norm in e1_norm:
        len_ratio = min(len(e1_norm), len(e2_norm)) / max(len(e1_norm), len(e2_norm))
        substring_sim = 0.9 * len_ratio

    # 5) Lightweight type-check to avoid cross-type merges
    type_indicators = {
        'person': {'president', 'minister', 'chancellor', 'commissioner', 'prime', 'secretary', 'ambassador', 'chair', 'director'},
        'org': {'union', 'organization', 'committee', 'council', 'parliament', 'commission', 'agency', 'institution', 'authority'},
        'policy': {'act', 'law', 'regulation', 'directive', 'treaty', 'agreement', 'protocol', 'convention', 'charter', 'constitution', 'bill', 'code'},
        'geo': {'country', 'nation', 'state', 'republic', 'kingdom', 'city', 'region', 'province', 'territory', 'district'}
    }
    def _type(ts):
        for t, inds in type_indicators.items():
            if ts.intersection(inds):
                return t
        return 'unknown'

    t1 = _type(tokens1)
    t2 = _type(tokens2)
    if t1 != 'unknown' and t2 != 'unknown' and t1 != t2:
        return 0.0

    # 6) Take the strongest reliable signal
    return max(acronym_sim, string_sim, token_sim, substring_sim)

def identify_person_entities(entities: List[str]) -> Set[str]:
    """Identify which entities are likely persons - MUCH MORE PRECISE"""
    persons = set()
    
    # Organization indicators that immediately disqualify as person
    org_words = {'union', 'parliament', 'council', 'committee', 'commission', 'organization', 
                 'institution', 'agency', 'department', 'ministry', 'body', 'authority', 
                 'association', 'federation', 'bank', 'company', 'corporation', 'group',
                 'alliance', 'league', 'party', 'government'}
    
    # Common first/last names to help identify real people
    common_names = {'mario', 'roberta', 'john', 'mary', 'michael', 'sarah', 'david', 'anna',
                   'draghi', 'metsola', 'smith', 'johnson', 'williams', 'brown', 'jones'}
    
    for entity in entities:
        words = entity.split()
        words_lower = [w.lower() for w in words]
        
        # RULE 1: If contains organization words, NOT a person
        if any(word in org_words for word in words_lower):
            continue
            
        # RULE 2: Title + Name pattern (e.g., "Prime Minister Draghi")
        title_words = {'president', 'prime', 'minister', 'chancellor', 'secretary', 
                      'director', 'commissioner', 'chair', 'chairman', 'head'}
        
        has_title = any(word in title_words for word in words_lower)
        has_likely_name = any(word in common_names for word in words_lower)
        
        if has_title and has_likely_name:
            persons.add(entity)
            continue
            
        # RULE 3: Just names (e.g., "Mario Draghi") - 2-3 words, all capitalized, likely names
        if 2 <= len(words) <= 3 and all(word[0].isupper() for word in words if word):
            # Check if words are common names or sound like names
            if any(word in common_names for word in words_lower):
                persons.add(entity)
                continue
                
            # Additional check: avoid geographic/organization patterns
            geo_words = {'european', 'american', 'italian', 'german', 'french', 'british'}
            if not any(word in geo_words for word in words_lower):
                # Only if it's 2 words and looks like "FirstName LastName"
                if len(words) == 2:
                    persons.add(entity)
    
    return persons

def _compute_local_kge_embeddings(
    relations: List[Tuple[str, str, str]],
    embedding_dim: Optional[int] = None,
    lr: Optional[float] = None,
    num_epochs: Optional[int] = None,
    patience: Optional[int] = None,
    frequency: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Train a tiny, per-speech KGE on provided relations and return entity embeddings.

    No caching and no reuse: produces precise, context-local vectors for coreference/linking.
    """
    if not PYKEEN_AVAILABLE or not relations:
        return {}
    try:
        import pandas as _pd
        _emb_dim = embedding_dim or PYKEEN_CONFIG.get('embedding_dim', 32)
        _lr = lr or PYKEEN_CONFIG.get('learning_rate', 1e-2)
        _epochs = num_epochs or PYKEEN_CONFIG.get('num_epochs', 25)
        _pat = patience or PYKEEN_CONFIG.get('early_stopping_patience', 5)
        _freq = frequency or PYKEEN_CONFIG.get('early_stopping_frequency', 3)

        df = _pd.DataFrame(relations, columns=['head', 'relation', 'tail'])
        tf = TriplesFactory.from_labeled_triples(df[['head', 'relation', 'tail']].values)

        result = pipeline(
            model='DistMult',
            training=tf,
            validation=tf,
            testing=tf,
            model_kwargs=dict(embedding_dim=_emb_dim),
            optimizer_kwargs=dict(lr=_lr),
            training_kwargs=dict(num_epochs=_epochs, use_tqdm_batch=False, checkpoint_frequency=0),
            evaluation_kwargs=dict(use_tqdm=False),
            stopper='early',
            stopper_kwargs=dict(frequency=_freq, patience=_pat),
            random_seed=42
        )

        entity_to_id = tf.entity_to_id
        if not entity_to_id:
            return {}
        all_ids = torch.arange(len(entity_to_id))
        with torch.no_grad():
            embs = result.model.entity_representations[0](all_ids).cpu().numpy()
        id_to_entity = {idx: ent for ent, idx in entity_to_id.items()}
        return {id_to_entity[i]: embs[i].astype(float).tolist() for i in range(len(id_to_entity))}
    except Exception as e:
        logger.warning(f"Local KGE training failed: {e}")
        return {}

def predict_missing_links_with_pykeen(relations: List[Tuple[str, str, str]], entities: List[str]) -> List[Tuple[str, str, str]]:
    """
    Use global signals to propose missing identity links.
    MODIFIED: Uses GLOBAL_ENTITY_EMBEDDINGS (corpus-wide) + enhanced string similarity; no local per-speech training.
    """
    # Disabled: no IS_SAME_AS prediction; always return empty list
    return []

def resolve_coreferences_with_pykeen(relations: List[Tuple[str, str, str]], entities: List[str]) -> Dict[str, Any]:
    """
    Enhanced coreference resolution with PyKEEN link prediction.
    1. Predict missing identity links between persons and titles
    2. Apply rule-based coreference resolution
    3. Return both raw and enhanced versions
    """
    if not PYKEEN_AVAILABLE or not relations:
        return {
            'raw_relations': relations,
            'cleaned_relations': relations,
            'predicted_links': [],
            'entity_mappings': {},
            'merged_entities': [],
            'debug_log': []
        }
    
    try:
        # Local debug log that mirrors important terminal logs
        debug_log: List[str] = []
        def _log(msg: str, level: str = "info") -> None:
            debug_log.append(msg)
            if level == "debug":
                logger.debug(msg)
            elif level == "warning":
                logger.warning(msg)
            else:
                logger.info(msg)

        # Step 1: Use PyKEEN to predict missing identity links
        predicted_links = predict_missing_links_with_pykeen(relations, entities)
        if predicted_links:
            _log(f"        PyKEEN predicted {len(predicted_links)} identity links")
        
        # Step 2: Combine original relations with predicted links
        enhanced_relations = relations + predicted_links
        
        # Step 3: Build similarity matrix for entities  
        unique_entities = list(set([rel[0] for rel in enhanced_relations] + [rel[2] for rel in enhanced_relations]))
        entity_to_idx = {entity: idx for idx, entity in enumerate(unique_entities)}
        
        # Compute pairwise similarities
        similarity_matrix = np.zeros((len(unique_entities), len(unique_entities)))
        for i, entity1 in enumerate(unique_entities):
            for j, entity2 in enumerate(unique_entities):
                if i != j:
                    similarity_matrix[i][j] = compute_entity_similarity(entity1, entity2)
        
        # Step 4: Skip boosting similarity via identity links (disabled)
        
        # Identify person entities for special handling
        person_entities = identify_person_entities(unique_entities)
        
        # Find potential coreferences (similarity > threshold)
        coreference_threshold = PYKEEN_CONFIG['coreference_threshold']
        entity_mappings = {}
        merged_groups = []
        
        _log(f"        Coreference resolution with threshold {coreference_threshold}")
        _log(f"        Analyzing {len(unique_entities)} entities for merging...")
        
        processed = set()
        for i, entity1 in enumerate(unique_entities):
            if entity1 in processed:
                continue
                
            # Find all entities similar to this one
            similar_entities = [entity1]
            for j, entity2 in enumerate(unique_entities):
                if i != j and entity2 not in processed:
                    sim_score = similarity_matrix[i][j]
                    if sim_score > coreference_threshold:
                        similar_entities.append(entity2)
                        _log(f"        MERGING: '{entity1}' <-> '{entity2}' (similarity: {sim_score:.3f})")
                    elif sim_score > 0.1:  # Debug: show what we're NOT merging
                        logger.debug(f"        NOT merging: '{entity1}' <-> '{entity2}' (similarity: {sim_score:.3f} < {coreference_threshold})")
            
            if len(similar_entities) > 1:
                # Optional guard: require strong base string similarity for any pair to be merged,
                # unless there is a clear acronym relation
                def _acronym_ok(a: str, b: str) -> bool:
                    t1 = a.lower().split()
                    t2 = b.lower().split()
                    if len(t1) == 1 and len(t1[0]) > 1 and len(t2) > 1:
                        acr = ''.join(w[0] for w in t2 if w and w[0].isalpha())
                        return t1[0] == acr
                    if len(t2) == 1 and len(t2[0]) > 1 and len(t1) > 1:
                        acr = ''.join(w[0] for w in t1 if w and w[0].isalpha())
                        return t2[0] == acr
                    return False

                def _passes_string_gate(group: list) -> bool:
                    from difflib import SequenceMatcher as _SM
                    min_required = PYKEEN_CONFIG.get('string_merge_min', 0.85)
                    for a in group:
                        for b in group:
                            if a == b:
                                continue
                            if _acronym_ok(a, b):
                                return True
                            if _SM(None, a.lower(), b.lower()).ratio() >= min_required:
                                return True
                    return False

                if not _passes_string_gate(similar_entities):
                    processed.add(entity1)
                    continue
                # Choose canonical entity (prefer person names over titles, longer names over shorter)
                canonical = similar_entities[0]
                for entity in similar_entities:
                    if entity in person_entities and len(entity.split()) >= 2:
                        # Prefer full person names
                        canonical = entity
                        break
                
                # If no clear person name, prefer longer, more descriptive entities
                if canonical == similar_entities[0]:
                    canonical = max(similar_entities, key=lambda x: (len(x.split()), len(x)))
                
                _log(f"        CREATED MERGED GROUP: {similar_entities} -> canonical: '{canonical}'")
                
                # Create mappings
                for entity in similar_entities:
                    entity_mappings[entity] = canonical
                    processed.add(entity)
                
                merged_groups.append({
                    'canonical': canonical,
                    'aliases': similar_entities,
                    'type': 'PERSON' if canonical in person_entities else determine_entity_type(canonical),
                    'predicted_via_pykeen': any((s, r, t) for s, r, t in predicted_links 
                                              if (s in similar_entities and t in similar_entities))
                })
            else:
                processed.add(entity1)
        
        # Apply mappings to create cleaned relations (identity links excluded)
        cleaned_relations = []
        for source, relation_type, target in enhanced_relations:
            if relation_type == "IS_SAME_AS":
                continue
            clean_source = entity_mappings.get(source, source)
            clean_target = entity_mappings.get(target, target)
            
            # Avoid self-loops unless they're meaningful
            if clean_source != clean_target:
                cleaned_relations.append((clean_source, relation_type, clean_target))
        
        # Remove duplicate relations
        cleaned_relations = list(set(cleaned_relations))
        
        _log(f"        Enhanced resolution: {len(relations)} -> {len(cleaned_relations)} relations")
        _log(f"        Predicted 0 identity links (disabled)")
        _log(f"        Merged {len(merged_groups)} entity groups")
        
        return {
            'raw_relations': relations,
            'cleaned_relations': cleaned_relations,
            'predicted_links': predicted_links,
            'entity_mappings': entity_mappings,
            'merged_entities': merged_groups,
            'coreference_threshold': coreference_threshold,
            'debug_log': debug_log
        }
        
    except Exception as e:
        logger.warning(f"        Enhanced coreference resolution failed: {str(e)}")
        return {
            'raw_relations': relations,
            'cleaned_relations': relations,
            'predicted_links': [],
            'entity_mappings': {},
            'merged_entities': [],
            'debug_log': [f"Error: {str(e)}"]
        }


# =============================================================================
# PER-SPEECH ENTITY-RELATION SUBGRAPH EMBEDDINGS (PyKEEN)
# =============================================================================

def _get_chunks_for_speech(graph: nx.DiGraph, speech_id: str) -> List[str]:
    """Return chunk node ids belonging to a given speech via HAS_CHUNK."""
    if not graph.has_node(speech_id):
        return []
    chunk_ids: List[str] = []
    for neighbor in graph.neighbors(speech_id):
        edge_data = graph.get_edge_data(speech_id, neighbor) or {}
        if edge_data.get('label') == 'HAS_CHUNK' and graph.nodes[neighbor].get('node_type') == 'CHUNK':
            chunk_ids.append(neighbor)
    return chunk_ids

def _get_entities_from_chunks(graph: nx.DiGraph, chunk_ids: List[str]) -> Set[str]:
    """Return entity ids connected from chunks via HAS_ENTITY."""
    entities: Set[str] = set()
    for chunk_id in chunk_ids:
        if not graph.has_node(chunk_id):
            continue
        for neighbor in graph.neighbors(chunk_id):
            edge_data = graph.get_edge_data(chunk_id, neighbor) or {}
            if (edge_data.get('label') == 'HAS_ENTITY' and 
                graph.nodes.get(neighbor, {}).get('node_type') == 'ENTITY_CONCEPT'):
                entities.add(neighbor)
    return entities

def _get_speaker_for_speech(graph: nx.DiGraph, speech_id: str) -> Optional[str]:
    """Find the speaker node that has HAS_SPEECH edge to the speech."""
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('node_type') == 'SPEAKER' and graph.has_edge(node_id, speech_id):
            edge = graph.get_edge_data(node_id, speech_id) or {}
            if edge.get('label') == 'HAS_SPEECH':
                return node_id
    return None

def _collect_triples_for_speech_entity_subgraph(graph: nx.DiGraph, speech_id: str) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
    """Collect relation triples restricted to entities appearing in this speech.
    Returns (triples, node_ids_involved). Speaker nodes and SPEAKER_MENTIONS edges are excluded.
    """
    chunk_ids = _get_chunks_for_speech(graph, speech_id)
    entity_ids = _get_entities_from_chunks(graph, chunk_ids)
    triples: List[Tuple[str, str, str]] = []

    if not entity_ids:
        return [], set()

    # Collect entity-entity relations among these entities from the global entity_relation layer
    for source, target, data in graph.edges(data=True):
        if source in entity_ids and target in entity_ids:
            if data.get('graph_type') == 'entity_relation':
                relation_type = data.get('relation_type') or data.get('label') or 'RELATED_TO'
                triples.append((str(source), str(relation_type), str(target)))

    node_set: Set[str] = set(entity_ids)
    return triples, node_set

def _compute_pykeen_embeddings_for_triples(triples: List[Tuple[str, str, str]]) -> Dict[str, List[float]]:
    """Deprecated: replaced by global embedding cache reuse. Kept for backward compatibility."""
    # Prefer global cache; fall back to empty dict to avoid per-speech training.
    return {}

def _compute_graph_stats(subgraph: nx.Graph) -> Dict[str, Any]:
    """Compute basic stats for an entity-relation subgraph."""
    stats: Dict[str, Any] = {
        'nodes': subgraph.number_of_nodes(),
        'edges': subgraph.number_of_edges(),
        'density': nx.density(subgraph) if subgraph.number_of_nodes() > 0 else 0.0,
        'avg_degree': (sum(dict(subgraph.degree()).values()) / subgraph.number_of_nodes()) if subgraph.number_of_nodes() > 0 else 0.0,
        'relation_type_counts': {}
    }
    rel_counts: Dict[str, int] = {}
    for _, _, data in subgraph.edges(data=True):
        rel = data.get('relation_type') or data.get('label') or 'RELATED_TO'
        rel_counts[rel] = rel_counts.get(rel, 0) + 1
    stats['relation_type_counts'] = rel_counts
    return stats

async def generate_embeddings_for_each_speech(graph: nx.DiGraph, output_dir: str) -> None:
    """For each speech, build a speech-local entity-relation subgraph, train embeddings, and save outputs."""
    try:
        target_dir = os.path.join(output_dir, 'speech_graph_embeddings')
        os.makedirs(target_dir, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create speech_graph_embeddings directory: {e}")
        return

    speech_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'SPEECH']
    logger.info(f"Generating per-speech embeddings for {len(speech_nodes)} speeches...")

    for speech_id in speech_nodes:
        try:
            triples, node_ids = _collect_triples_for_speech_entity_subgraph(graph, speech_id)

            # Build a simple NetworkX subgraph to compute stats (edge attributes preserved)
            subgraph = nx.Graph()
            # Add nodes with original attributes when available
            for nid in node_ids:
                if graph.has_node(nid):
                    subgraph.add_node(nid, **graph.nodes[nid])
                else:
                    subgraph.add_node(nid)
            # Add edges that are part of the triples
            for h, r, t in triples:
                attrs = {'relation_type': r, 'graph_type': 'entity_relation'}
                subgraph.add_edge(h, t, **attrs)

            stats = _compute_graph_stats(subgraph)

            # Reuse global embeddings instead of per-speech training
            embeddings: Dict[str, List[float]] = {}
            for nid in node_ids:
                vec = graph.nodes.get(nid, {}).get('kge_embedding') or GLOBAL_ENTITY_EMBEDDINGS.get(str(nid))
                if isinstance(vec, list) and vec:
                    embeddings[nid] = vec
            if not embeddings:
                logger.info(f"No global embeddings available for {speech_id}; writing stats only (relations={len(triples)})")

            # Speech embeddings JSON saving removed per user request

            # Save stats JSON
            stats_path = os.path.join(target_dir, f"{speech_id}_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'speech_id': speech_id,
                    'stats': stats,
                    'has_embeddings': bool(embeddings),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False, default=json_serialize_dates)

        except Exception as e:
            logger.warning(f"Failed per-speech embedding generation for {speech_id}: {e}")

def extract_abstract_concepts_sync(text: str) -> List[str]:
    """Extract abstract concepts using pure grammatical and linguistic filtering"""
    try:
        tokens = word_tokenize(text.lower())
        tagged_tokens = pos_tag(tokens)
        
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()  # Fallback if stopwords not available
        
        concepts = set()
        
        for word, tag in tagged_tokens:
            # Skip stop words
            if word in stop_words:
                continue
            
            # Skip pronouns, determiners, and function words using POS tags
            if is_pronoun_or_determiner(word, tag):
                continue
            
            # Process only nouns
            wn_tag = get_wordnet_pos(tag)
            if wn_tag == wn.NOUN:
                lemma = lemmatizer.lemmatize(word, pos=wn_tag)
                
                # Apply grammatical and linguistic filtering
                if (is_grammatically_valid_concept(word, lemma, tag) and
                    not is_temporal_or_spatial_deictic(lemma) and
                    not is_modal_or_auxiliary_concept(lemma, tag) and
                    is_abstract_noun(lemma)):
                    concepts.add(lemma)
        
        return sorted(list(concepts))
    except Exception as e:
        logger.error(f"Error in concept extraction: {e}", exc_info=True)
        return []

def extract_entities_sync(text: str) -> List[str]:
    """EXACT match to original relation_extraction.py extract_entities function"""
    try:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        potential_entities = []
        current_entity_parts = []
        
        for word, tag in pos_tags:
            if tag == 'NNP':  # Proper noun, singular
                current_entity_parts.append(word)
            else:
                if current_entity_parts:
                    # Check if all parts of the collected entity are in WordNet as nouns
                    all_parts_in_wordnet = True
                    for part in current_entity_parts:
                        if not wn.synsets(part.lower(), pos=wn.NOUN):
                            all_parts_in_wordnet = False
                            break
                    
                    if all_parts_in_wordnet:
                        potential_entities.append(" ".join(current_entity_parts))
                    current_entity_parts = []
        
        # Check for any remaining entity at the end of the text
        if current_entity_parts:
            all_parts_in_wordnet = True
            for part in current_entity_parts:
                if not wn.synsets(part.lower(), pos=wn.NOUN):
                    all_parts_in_wordnet = False
                    break
            if all_parts_in_wordnet:
                potential_entities.append(" ".join(current_entity_parts))
        
        return potential_entities
    except Exception as e:
        logger.error(f"Error in entity extraction: {e}", exc_info=True)
        return []

async def extract_relations_with_llm_async(text: str, entities: List[str], abstract_concepts: List[str], max_retries: int = 3) -> List[Tuple[str, str, str]]:
    """
    Extract relations using the EXACT same approach as original relation_extraction.py but async.
    This matches the original quality while providing async benefits.
    """
    if not LLM_AVAILABLE or not llm:
        return []
        
    if not entities and not abstract_concepts:
        return []
    
    # Combine entities and abstract concepts for allowed nodes (EXACT match to original)
    allowed_nodes = list(set(entities + abstract_concepts))
    
    # Create the EXACT same custom prompt as the original
    prompt_template = """
            You are an expert knowledge graph extraction system. Your task is to extract relationships between the specified entities and concepts from the provided text.

            **FOCUS ENTITIES AND CONCEPTS:**
            {allowed_nodes}

            **EXTRACTION GUIDELINES:**
            1. Extract only relationships that are explicitly mentioned or clearly implied in the text
            2. Focus on relationships between the entities and concepts listed above
            3. Use clear, descriptive relationship types.

            **TEXT TO ANALYZE:**
            {text}

            Extract relationships that connect the specified entities and concepts. Be precise and only extract relationships that are clearly supported by the text.
            """
        
    # Format the prompt (EXACT match to original)
    formatted_prompt = prompt_template.format(
        allowed_nodes=", ".join(allowed_nodes),
        text=text
    )
    
    # Create prompt template (EXACT match to original)
    prompt = ChatPromptTemplate.from_messages([
        ("system", formatted_prompt)
    ])
    
    # Initialize LLMGraphTransformer with EXACT same parameters as original
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        prompt=prompt,
        strict_mode=True,
        node_properties=False,
        relationship_properties=False
    )

    # Add retry logic (EXACT match to original)
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            # Create document (EXACT match to original)
            document = Document(page_content=text)
            
            # Extract graph using async if available, otherwise sync (EXACT match to original approach)
            try:
                # Try async first
                graph_docs = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    transformer.convert_to_graph_documents, 
                    [document]
                )
            except:
                # Fallback to sync if async fails
                graph_docs = transformer.convert_to_graph_documents(documents=[document])
            
            if not graph_docs:
                return []
            
            # Extract relations as triplets (EXACT match to original)
            relations = []
            graph_doc = graph_docs[0]
            
            for relationship in graph_doc.relationships:
                source = relationship.source.id
                target = relationship.target.id
                relation_type = relationship.type
                
                relations.append((source, relation_type, target))
            
            return relations
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.info(f"      Retry {attempt + 1}/{max_retries} after error: {str(e)}")
                await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            else:
                logger.error(f"      Failed after {max_retries} attempts: {str(e)}", exc_info=True)
                return []
    
    return []

async def add_triplets_to_graph_async(deps: AgentDependencies, relations: List[Tuple[str, str, str]], source_chunk_id: str) -> Dict[str, Any]:
    """LEGACY (unused): Per-chunk enrichment helper retained for backward compatibility.

    Note: Enrichment, entity/coreference resolution, and PyKEEN-based consolidation now run once per speech
    via `enrich_graph_per_speech(...)` and `add_triplets_to_graph_for_speech(...)`. Do not call this function
    from new code; chunk-level remains extraction-only and stores raw results on the chunk node.

    Original behavior (unchanged): Add relation triplets to the graph as nodes and edges with proper metadata tracking.
    """
    
    if not relations:
        return None
    
    logger.info(f"      Adding {len(relations)} relation triplets from {source_chunk_id}")
    
    # Track processed entities to avoid duplicates within this chunk
    processed_entities = set()
    entities_added = []
    relations_added = []
    
    # First pass: Process all entities from relations (exact match to original)
    for source, relation_type, target in relations:
        # Process source entity
        if source not in processed_entities:
            processed_entities.add(source)
            source_id = source.replace(' ', '_').upper()
            entity_type = determine_entity_type(source)
            
            # Add source node if it doesn't exist
            if source_id not in deps.graph:
                deps.graph.add_node(source_id, 
                              node_type="ENTITY_CONCEPT",
                              graph_type="entity_relation",
                              name=source,
                              entity_type=entity_type,
                              extracted_from=[source_chunk_id])
                entities_added.append({
                    "id": source_id,
                    "name": source,
                    "type": entity_type
                })
            else:
                # Update extracted_from list for existing entity
                extracted_from = deps.graph.nodes[source_id].get('extracted_from', [])
                if source_chunk_id not in extracted_from:
                    extracted_from.append(source_chunk_id)
                    deps.graph.nodes[source_id]['extracted_from'] = extracted_from
            
            # Connect chunk to entity (HAS_ENTITY relationship)
            if not deps.graph.has_edge(source_chunk_id, source_id):
                deps.graph.add_edge(source_chunk_id, source_id, 
                             label="HAS_ENTITY", 
                             graph_type="lexical_graph")
        
        # Process target entity
        if target not in processed_entities:
            processed_entities.add(target)
            target_id = target.replace(' ', '_').upper()
            entity_type = determine_entity_type(target)
            
            # Add target node if it doesn't exist
            if target_id not in deps.graph:
                deps.graph.add_node(target_id, 
                              node_type="ENTITY_CONCEPT",
                              graph_type="entity_relation",
                              name=target,
                              entity_type=entity_type,
                              extracted_from=[source_chunk_id])
                entities_added.append({
                    "id": target_id,
                    "name": target,
                    "type": entity_type
                })
            else:
                # Update extracted_from list for existing entity
                extracted_from = deps.graph.nodes[target_id].get('extracted_from', [])
                if source_chunk_id not in extracted_from:
                    extracted_from.append(source_chunk_id)
                    deps.graph.nodes[target_id]['extracted_from'] = extracted_from
            
            # Connect chunk to entity (HAS_ENTITY relationship)
            if not deps.graph.has_edge(source_chunk_id, target_id):
                deps.graph.add_edge(source_chunk_id, target_id, 
                             label="HAS_ENTITY", 
                             graph_type="lexical_graph")
    
    # Second pass: Process all relations (exact match to original)
    for source, relation_type, target in relations:
        source_id = source.replace(' ', '_').upper()
        target_id = target.replace(' ', '_').upper()
        
        # Add edge between entities/concepts
        if not deps.graph.has_edge(source_id, target_id):
            deps.graph.add_edge(source_id, target_id, 
                          label=relation_type,
                          graph_type="entity_relation",
                          relation_type=relation_type,
                          source_name=source,
                          target_name=target,
                          extracted_from=[source_chunk_id])
            
            relations_added.append({
                "source": source,
                "target": target,
                "type": relation_type,
                "source_id": source_id,
                "target_id": target_id
            })
        else:
            # Update extracted_from list for existing edge
            existing_edge = deps.graph.edges[source_id, target_id]
            extracted_from = existing_edge.get('extracted_from', [])
            if source_chunk_id not in extracted_from:
                extracted_from.append(source_chunk_id)
                deps.graph.edges[source_id, target_id]['extracted_from'] = extracted_from
    
    # Third pass: Create SPEAKER_MENTIONS relationships
    # Find the speaker for this chunk by traversing: CHUNK -> SPEECH -> SPEAKER
    try:
        # Get speech ID from chunk ID - fix double SPEECH prefix issue
        # CHUNK_SPEECH_DOC_9_... -> SPEECH_DOC_9_...
        chunk_without_prefix = source_chunk_id.replace('CHUNK_SPEECH_', '')
        speech_id = chunk_without_prefix.rsplit('_', 1)[0]  # Remove chunk number
        speech_id = f"SPEECH_{speech_id}"
        
        # Find speaker who gave this speech
        speaker_id = None
        for node_id, node_data in deps.graph.nodes(data=True):
            if (node_data.get('node_type') == 'SPEAKER' and 
                deps.graph.has_edge(node_id, speech_id)):
                speaker_id = node_id
                break
        
        if speaker_id:
            # Create SPEAKER_MENTIONS relationships for all entities in this chunk
            for source, relation_type, target in relations:
                source_id = source.replace(' ', '_').upper()
                target_id = target.replace(' ', '_').upper()
                
                # Add SPEAKER_MENTIONS relationship from speaker to source entity
                if not deps.graph.has_edge(speaker_id, source_id):
                    deps.graph.add_edge(speaker_id, source_id,
                                          label="SPEAKER_MENTIONS",
                                          graph_type="entity_relation",
                                          relation_type="SPEAKER_MENTIONS",
                                          mentioned_in_chunk=source_chunk_id)
                
                # Add SPEAKER_MENTIONS relationship from speaker to target entity
                if not deps.graph.has_edge(speaker_id, target_id):
                    deps.graph.add_edge(speaker_id, target_id,
                                          label="SPEAKER_MENTIONS", 
                                          graph_type="entity_relation",
                                          relation_type="SPEAKER_MENTIONS",
                                          mentioned_in_chunk=source_chunk_id)
                
            logger.info(f"        Added SPEAKER_MENTIONS relationships from {speaker_id} to entities")
    except Exception as e:
        logger.warning(f"        Could not create SPEAKER_MENTIONS relationships: {str(e)}")
    
    # Update chunk metadata with extraction results  
    if source_chunk_id in deps.graph:
        chunk_node = deps.graph.nodes[source_chunk_id]
        chunk_node['graph_entities_added'] = entities_added
        chunk_node['graph_relations_added'] = relations_added
    
    logger.info(f"        Added {len(entities_added)} entities and {len(relations_added)} relations")
    return {
        "entities": entities_added,
        "relations": relations_added
    }

def json_serialize_dates(obj):
    """Custom JSON serializer for datetime, numpy, and common non-JSON types"""
    # datetime-like
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    # numpy scalar types
    try:
        import numpy as _np  # use local import to avoid shadowing
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
        # numpy arrays
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    # sets -> lists
    if isinstance(obj, set):
        return list(obj)
    # bytes -> utf-8 string best-effort
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode('utf-8', errors='replace')
        except Exception:
            return str(obj)
    # fallback to str to avoid hard failure on exotic objects
    return str(obj)

# INTEGRATED COMMUNITY DETECTION FUNCTIONALITY

async def build_semantic_kg_with_communities(
    input_dir: str = None,
    output_dir: str = None,
    save_files: bool = True,
    config_overrides: Dict[str, Any] = None
) -> SemanticKGResults:
    """
    Build semantic knowledge graph with integrated community detection.
    
    This function combines the semantic entity extraction approach from pure_semantic_kg.py
    with community detection to produce output compatible with topic_summarizer.py
    
    Args:
        input_dir: Override input directory
        output_dir: Override output directory  
        save_files: Whether to save output files
        config_overrides: Dictionary of configuration overrides (e.g., {'speech_limit': 100})
    """
    
    # Use shared configuration with overrides
    if config_overrides is None:
        config_overrides = {}
    
    config = get_shared_config(**config_overrides)
    
    if input_dir is None:
        input_dir = config.get('input_dir')
    if output_dir is None:
        output_dir = config.get('output_dir')
    
    # Update global variables from config
    global SPEECH_LIMIT, USE_SPEECH_LIMIT, MAX_CONCURRENT_EXTRACTIONS
    SPEECH_LIMIT = config.get('speech_limit')
    USE_SPEECH_LIMIT = config.get('use_speech_limit') 
    MAX_CONCURRENT_EXTRACTIONS = config.get('max_concurrent_extractions')
    
    logger.info("Starting Semantic Knowledge Graph Generation with Community Detection...")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    # Step 1: Build base semantic knowledge graph
    logger.info("Phase 1: Building semantic knowledge graph...")
    logger.info(f"Using input directory: {input_dir}")
    logger.info(f"Speech limit: {SPEECH_LIMIT} (enabled: {USE_SPEECH_LIMIT})")
    
    # Initialize dependencies
    base_graph = nx.DiGraph()
    speakers_seen = set()
    
    deps = AgentDependencies(
        graph=base_graph,
        nlp_model=nlp,
        speakers_seen=speakers_seen
    )
    
    # Pass input directory to build_knowledge_graph
    kg_result = await build_knowledge_graph(deps, input_dir)
    
    logger.info(f"Base graph completed: {base_graph.number_of_nodes()} nodes, {base_graph.number_of_edges()} edges")
    
    # Train one global KGE over all entity relations and cache vectors on nodes
    try:
        _train_and_cache_global_kge(base_graph, output_dir)
    except Exception as e:
        logger.warning(f"Global KGE training failed: {e}")

    # Generate per-speech overlays using global embeddings (no training per speech)
    try:
        await generate_embeddings_for_each_speech(base_graph, output_dir)
    except Exception as e:
        logger.warning(f"Per-speech embedding overlay failed: {e}")

    # Step 2: Extract entity-relation subgraph for community detection
    logger.info("Phase 2: Extracting entity-relation subgraph...")
    
    entity_graph = extract_entity_relation_subgraph(base_graph)
    logger.info(f"Entity graph extracted: {entity_graph.number_of_nodes()} nodes, {entity_graph.number_of_edges()} edges")
    
    if entity_graph.number_of_nodes() == 0:
        logger.warning("No entity nodes found in the graph. Cannot perform community detection.")
        # Return minimal results
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return SemanticKGResults(
            base_graph=base_graph,
            entity_graph=entity_graph,
            communities={},
            subcommunities={},
            enhanced_graph=base_graph,
            statistics=_generate_statistics(base_graph, entity_graph, {}, {}),
            processing_time=processing_time,
            community_quality=CommunityQualityMetrics(0, 0, 0, 0, 0, 0, 0, False)
        )
    
    # Step 3: Perform community detection
    logger.info("Phase 3: Detecting communities...")
    
    # Exclude SPEAKER nodes from community detection by using an entity-only subgraph
    entity_only_nodes = [n for n, d in entity_graph.nodes(data=True) if d.get('node_type') == 'ENTITY_CONCEPT']
    entity_only_graph = entity_graph.subgraph(entity_only_nodes).copy()
    detector = CommunityDetector()
    communities = detector.detect_communities(entity_only_graph)
    
    logger.info(f"Communities detected: {len(set(communities.values()))} communities")
    
    # Step 4: Detect subcommunities
    logger.info("Phase 4: Detecting subcommunities...")
    
    subcommunities: Dict[str, Tuple[int, int]] = {}
    if len(set(communities.values())) > 0:
        cd_conf = config.get('community_detection') or {}
        sub_map = detector.detect_subcommunities_leiden(
            entity_only_graph,
            communities,
            min_sub_size=cd_conf.get('min_subcomm_size', 2),
            sub_resolution_min=cd_conf.get('sub_resolution_min', 0.7),
            sub_resolution_max=cd_conf.get('sub_resolution_max', 1.3),
            sub_resolution_steps=cd_conf.get('sub_resolution_steps', 7),
            max_depth=cd_conf.get('sub_max_depth', 1)
        )
        subcommunities = sub_map
    
    logger.info(f"Subcommunities detected: {len(set([sid for (_, sid) in subcommunities.values()]))} subcommunities across parents")
    
    # Step 5: Evaluate community quality
    community_quality = evaluate_community_quality(base_graph, communities)
    
    # Step 6: Add community attributes to the original graph
    logger.info("Phase 5: Adding community attributes to graph...")
    
    enhanced_graph = add_enhanced_community_attributes_to_graph(
        base_graph.copy(), communities, subcommunities
    )
    
    logger.info("Community attributes added to graph")
    
    # Step 6: Generate RAG embeddings
    logger.info("Phase 6: Generating RAG embeddings...")
    embeddings = generate_rag_embeddings(enhanced_graph)
    
    # Step 7: Generate comprehensive statistics
    statistics = _generate_statistics(base_graph, entity_graph, communities, subcommunities)
    statistics["embeddings_generated"] = len(embeddings)
    
    # Step 8: Calculate centrality measures for entity-relation nodes
    logger.info("Phase 8: Calculating centrality measures for entity-relation nodes...")
    centrality_results = calculate_entity_relation_centrality_measures(enhanced_graph)
    statistics["centrality_analysis"] = centrality_results
    
    # Step 8b: Save and plot top centrality measures
    if centrality_results and not centrality_results.get('error'):
        centrality_plots = save_and_plot_top_centrality_measures(centrality_results, output_dir)
        statistics.setdefault("output_files", {})
        statistics["output_files"]["centrality_plots"] = centrality_plots

    # Step 9: Summaries for communities (claims already generated earlier)
    try:
        summary_results = generate_community_and_subcommunity_summaries_from_claims(enhanced_graph)
        statistics["community_summaries_written"] = len(summary_results)
    except Exception as e:
        logger.warning(f"Community summary generation failed: {e}")
        statistics["community_summaries_error"] = str(e)
    
    # Step 10: Compare community summaries
    try:
        comparison_results = generate_community_summary_comparison(enhanced_graph, output_dir)
        statistics["summary_comparison"] = comparison_results
    except Exception as e:
        logger.warning(f"Community summary comparison failed: {e}")
        statistics["summary_comparison_error"] = str(e)
    
    # Derive and save the most distinct (low-similarity) topics for follow-up analysis
    try:
        distinct_path = save_distinct_low_similarity_topics(
            enhanced_graph,
            statistics.get("summary_comparison", {}),
            output_dir,
            threshold_percentile=10
        )
        if distinct_path:
            statistics.setdefault("output_files", {})
            statistics["output_files"]["distinct_topic_summaries"] = distinct_path
    except Exception:
        pass
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    logger.info(f"Total processing time: {processing_time:.2f} seconds")
    
    # Step 11: Save results if requested
    if save_files:
        await _save_results(
            enhanced_graph, entity_graph, communities, subcommunities, 
            statistics, processing_time, output_dir, embeddings,
            centrality_results, community_quality
        )
    
    return SemanticKGResults(
        base_graph=base_graph,
        entity_graph=entity_graph,
        communities=communities,
        subcommunities=subcommunities,
        enhanced_graph=enhanced_graph,
        statistics=statistics,
        processing_time=processing_time,
        community_quality=community_quality
    )

def _generate_statistics(
    base_graph: nx.DiGraph,
    entity_graph: nx.Graph,
    communities: Dict[str, int],
    subcommunities: Dict[str, int]
) -> Dict[str, Any]:
    """Generate comprehensive statistics for the semantic knowledge graph"""
    
    # Base graph statistics
    base_stats = {
        "total_nodes": base_graph.number_of_nodes(),
        "total_edges": base_graph.number_of_edges(),
        "documents": len([n for n, d in base_graph.nodes(data=True) if d.get('node_type') == 'DOCUMENT']),
        "speakers": len([n for n, d in base_graph.nodes(data=True) if d.get('node_type') == 'SPEAKER']),
        "speeches": len([n for n, d in base_graph.nodes(data=True) if d.get('node_type') == 'SPEECH']),
        "chunks": len([n for n, d in base_graph.nodes(data=True) if d.get('node_type') == 'CHUNK']),
        "entities_concepts": len([n for n, d in base_graph.nodes(data=True) if d.get('node_type') == 'ENTITY_CONCEPT'])
    }
    
    # Entity graph statistics
    entity_stats = {
        "entity_nodes": entity_graph.number_of_nodes(),
        "entity_edges": entity_graph.number_of_edges(),
        "density": nx.density(entity_graph) if entity_graph.number_of_nodes() > 0 else 0,
        "avg_degree": sum(dict(entity_graph.degree()).values()) / entity_graph.number_of_nodes() if entity_graph.number_of_nodes() > 0 else 0
    }
    
    # Community statistics
    community_stats = {
        "num_communities": len(set(communities.values())) if communities else 0,
        "num_subcommunities": len(set(subcommunities.values())) if subcommunities else 0,
        "community_sizes": {},
        "subcommunity_sizes": {}
    }
    
    if communities:
        # Calculate community sizes
        community_counts = {}
        for entity, comm_id in communities.items():
            community_counts[comm_id] = community_counts.get(comm_id, 0) + 1
        
        if community_counts:
            community_stats["community_sizes"] = {
                "min": min(community_counts.values()),
                "max": max(community_counts.values()),
                "avg": sum(community_counts.values()) / len(community_counts),
                "distribution": community_counts
            }
    
    if subcommunities:
        # Calculate subcommunity sizes
        subcomm_counts = {}
        for entity, subcomm_id in subcommunities.items():
            # Normalize subcommunity key to a string to ensure JSON-safe dict keys
            if isinstance(subcomm_id, tuple) and len(subcomm_id) == 2:
                key = f"{subcomm_id[0]}_{subcomm_id[1]}"
            else:
                key = str(subcomm_id)
            subcomm_counts[key] = subcomm_counts.get(key, 0) + 1
        
        if subcomm_counts:
            community_stats["subcommunity_sizes"] = {
                "min": min(subcomm_counts.values()),
                "max": max(subcomm_counts.values()),
                "avg": sum(subcomm_counts.values()) / len(subcomm_counts),
                "distribution": subcomm_counts
            }
    
    # Entity type distribution
    entity_types = {}
    for node_id, node_data in base_graph.nodes(data=True):
        if node_data.get('node_type') == 'ENTITY_CONCEPT':
            entity_type = node_data.get('entity_type', 'UNKNOWN')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    # Graph structure analysis
    structure_stats = {
        "lexical_graph_nodes": len([n for n, d in base_graph.nodes(data=True) if d.get('graph_type') == 'lexical_graph']),
        "entity_relation_nodes": len([n for n, d in base_graph.nodes(data=True) if d.get('graph_type') == 'entity_relation']),
        "lexical_graph_edges": len([e for e in base_graph.edges(data=True) if e[2].get('graph_type') == 'lexical_graph']),
        "entity_relation_edges": len([e for e in base_graph.edges(data=True) if e[2].get('graph_type') == 'entity_relation'])
    }
    
    return {
        "base_graph_stats": base_stats,
        "entity_graph_stats": entity_stats,
        "community_stats": community_stats,
        "entity_type_distribution": entity_types,
        "structure_stats": structure_stats,
        "methodology": "semantic_with_communities",
        "timestamp": datetime.now().isoformat()
    }

def generate_executive_summary_markdown(
    base_graph: nx.DiGraph,
    entity_graph: nx.Graph,
    communities: Dict[str, int],
    subcommunities: Dict[str, int],
    centrality_results: Dict[str, Any],
    community_quality: CommunityQualityMetrics,
    statistics: Dict[str, Any],
    processing_time: float,
    output_dir: str
) -> str:
    """
    Generate a comprehensive executive summary markdown document with all key metrics and findings.
    
    Args:
        base_graph: The complete knowledge graph
        entity_graph: Entity-only subgraph
        communities: Community assignments
        subcommunities: Subcommunity assignments
        centrality_results: Centrality calculation results
        community_quality: Community quality metrics
        statistics: General statistics
        processing_time: Total processing time
        output_dir: Output directory for the markdown file
        
    Returns:
        Path to the generated markdown file
    """
    logger.info("Generating executive summary markdown...")
    
    # Calculate additional metrics
    entity_types = {}
    for node_id, node_data in base_graph.nodes(data=True):
        if node_data.get('node_type') == 'ENTITY_CONCEPT':
            entity_type = node_data.get('entity_type', 'UNKNOWN')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    # Get top entities by centrality
    top_entities = {}
    if centrality_results.get('top_nodes'):
        for measure, nodes in centrality_results['top_nodes'].items():
            top_entities[measure] = [(node_id, score, base_graph.nodes[node_id].get('name', node_id)) 
                                   for node_id, score in nodes[:5]]
    
    # Calculate graph connectivity metrics
    connectivity_stats = {
        'is_connected': nx.is_weakly_connected(base_graph),
        'num_components': len(list(nx.weakly_connected_components(base_graph))),
        'largest_component_size': len(max(nx.weakly_connected_components(base_graph), key=len)) if base_graph.number_of_nodes() > 0 else 0
    }
    
    # Generate markdown content
    markdown_content = f"""# Knowledge Graph Executive Summary

## Overview
This document provides a comprehensive analysis of the semantic knowledge graph generated from parliamentary proceedings, including key metrics, community structure, and entity importance measures.

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Processing Time:** {processing_time:.2f} seconds

---

## Graph Structure Metrics

### Basic Statistics
- **Total Nodes:** {base_graph.number_of_nodes():,}
- **Total Edges:** {base_graph.number_of_edges():,}
- **Graph Density:** {nx.density(base_graph):.6f}
- **Average Degree:** {sum(dict(base_graph.degree()).values()) / base_graph.number_of_nodes() if base_graph.number_of_nodes() > 0 else 0:.2f}

### Node Type Distribution
- **Documents:** {statistics['base_graph_stats']['documents']:,}
- **Speakers:** {statistics['base_graph_stats']['speakers']:,}
- **Speeches:** {statistics['base_graph_stats']['speeches']:,}
- **Text Chunks:** {statistics['base_graph_stats']['chunks']:,}
- **Entities/Concepts:** {statistics['base_graph_stats']['entities_concepts']:,}

### Entity Type Distribution
"""
    
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        markdown_content += f"- **{entity_type}:** {count:,}\n"
    
    markdown_content += f"""
### Connectivity Analysis
- **Graph Connected:** {'Yes' if connectivity_stats['is_connected'] else 'No'}
- **Connected Components:** {connectivity_stats['num_components']:,}
- **Largest Component Size:** {connectivity_stats['largest_component_size']:,} nodes

---

## Entity Graph Analysis

### Entity-Only Subgraph
- **Entity Nodes:** {entity_graph.number_of_nodes():,}
- **Entity Edges:** {entity_graph.number_of_edges():,}
- **Entity Graph Density:** {nx.density(entity_graph):.6f}
- **Average Entity Degree:** {sum(dict(entity_graph.degree()).values()) / entity_graph.number_of_nodes() if entity_graph.number_of_nodes() > 0 else 0:.2f}

### Scale-Free Properties
"""
    
    # Add scale-free analysis if available
    if 'scale_free_analysis' in statistics:
        sf_analysis = statistics['scale_free_analysis']
        markdown_content += f"""
- **Power-Law Exponent (γ):** {sf_analysis.get('gamma', 'N/A')}
- **R² Goodness of Fit:** {sf_analysis.get('r_squared', 'N/A')}
- **Scale-Free Assessment:** {sf_analysis.get('is_scale_free', 'N/A')}
"""
    else:
        markdown_content += "- **Scale-Free Analysis:** Not available\n"
    
    markdown_content += f"""
---

## Centrality Measures

### Top Entities by Centrality

"""
    
    # Add top entities for each centrality measure
    for measure, entities in top_entities.items():
        markdown_content += f"#### {measure.replace('_', ' ').title()} Centrality\n"
        for i, (node_id, score, name) in enumerate(entities, 1):
            markdown_content += f"{i}. **{name}** (Score: {score:.4f})\n"
        markdown_content += "\n"
    
    # Add centrality statistics
    if centrality_results.get('statistics'):
        markdown_content += "### Centrality Statistics\n\n"
        for measure, stats in centrality_results['statistics'].items():
            markdown_content += f"#### {measure.replace('_', ' ').title()}\n"
            markdown_content += f"- **Mean:** {stats['mean']:.4f}\n"
            markdown_content += f"- **Std Dev:** {stats['std']:.4f}\n"
            markdown_content += f"- **Min:** {stats['min']:.4f}\n"
            markdown_content += f"- **Max:** {stats['max']:.4f}\n"
            markdown_content += f"- **Median:** {stats['median']:.4f}\n\n"
    
    markdown_content += f"""
---

## Community Structure Analysis

### Community Detection Results
- **Number of Communities:** {statistics['community_stats']['num_communities']:,}
- **Number of Subcommunities:** {statistics['community_stats']['num_subcommunities']:,}

### Community Size Distribution
"""
    
    if statistics['community_stats']['community_sizes']:
        comm_sizes = statistics['community_stats']['community_sizes']
        markdown_content += f"""
- **Minimum Community Size:** {comm_sizes['min']:,}
- **Maximum Community Size:** {comm_sizes['max']:,}
- **Average Community Size:** {comm_sizes['avg']:.2f}
"""
    
    if statistics['community_stats']['subcommunity_sizes']:
        subcomm_sizes = statistics['community_stats']['subcommunity_sizes']
        markdown_content += f"""
- **Minimum Subcommunity Size:** {subcomm_sizes['min']:,}
- **Maximum Subcommunity Size:** {subcomm_sizes['max']:,}
- **Average Subcommunity Size:** {subcomm_sizes['avg']:.2f}
"""
    
    markdown_content += f"""
### Community Quality Metrics
- **Modularity:** {community_quality.modularity:.4f}
- **Size Variance:** {community_quality.size_variance:.2f}
- **Size Ratio (Max/Min):** {community_quality.size_ratio:.2f}
- **Text Length Variance:** {community_quality.text_length_variance:.2f}
- **Optimal for Summarization:** {'Yes' if community_quality.optimal_for_summarization else 'No'}

---

## Processing Performance

### Extraction Statistics
- **Total Processing Time:** {processing_time:.2f} seconds
- **Entities per Second:** {statistics['base_graph_stats']['entities_concepts'] / processing_time:.2f}
- **Relations per Second:** {statistics['base_graph_stats']['total_edges'] / processing_time:.2f}

### Quality Indicators
- **Entity Coverage:** {statistics['base_graph_stats']['entities_concepts'] / statistics['base_graph_stats']['chunks']:.2f} entities per chunk
- **Relation Density:** {statistics['base_graph_stats']['total_edges'] / statistics['base_graph_stats']['entities_concepts']:.2f} relations per entity

---

## Key Insights

### Graph Structure
- The knowledge graph contains {base_graph.number_of_nodes():,} nodes and {base_graph.number_of_edges():,} edges
- Graph density of {nx.density(base_graph):.6f} indicates a sparse but well-connected structure
- {connectivity_stats['num_components']} connected components suggest multiple thematic clusters

### Entity Importance
"""
    
    # Add insights about top entities
    if top_entities.get('pagerank'):
        top_entity = top_entities['pagerank'][0]
        markdown_content += f"- **Most Central Entity:** {top_entity[2]} (PageRank: {top_entity[1]:.4f})\n"
    
    if top_entities.get('betweenness'):
        bridge_entity = top_entities['betweenness'][0]
        markdown_content += f"- **Key Bridge Entity:** {bridge_entity[2]} (Betweenness: {bridge_entity[1]:.4f})\n"
    
    markdown_content += f"""
### Community Quality
- Modularity of {community_quality.modularity:.4f} indicates {'strong' if community_quality.modularity > 0.3 else 'moderate'} community structure
- {'Optimal' if community_quality.optimal_for_summarization else 'Suboptimal'} community balance for summarization tasks
- Size ratio of {community_quality.size_ratio:.2f} shows {'balanced' if community_quality.size_ratio < 10 else 'imbalanced'} community distribution

### Processing Efficiency
- Achieved {statistics['base_graph_stats']['entities_concepts'] / processing_time:.2f} entities per second
- Generated {statistics['base_graph_stats']['total_edges'] / processing_time:.2f} relations per second
- Average of {statistics['base_graph_stats']['entities_concepts'] / statistics['base_graph_stats']['chunks']:.2f} entities per text chunk

---

## Recommendations

### For Further Analysis
1. **Focus on high-centrality entities** for topic modeling and summarization
2. **Investigate bridge entities** for cross-domain connections
3. **Analyze community summaries** for thematic coherence
4. **Examine entity type distributions** for domain coverage

### For Graph Optimization
1. **Consider community rebalancing** if size ratio > 10
2. **Investigate isolated components** for potential connections
3. **Validate entity type classifications** for accuracy
4. **Monitor processing efficiency** for scalability

---

*This executive summary was automatically generated from the knowledge graph analysis pipeline.*
"""
    
    # Save markdown file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    markdown_file = os.path.join(output_dir, f'knowledge_graph_executive_summary_{timestamp}.md')
    
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    logger.info(f"Executive summary saved to: {markdown_file}")
    return markdown_file

def analyze_scale_free_entity_graph(entity_graph: nx.Graph, output_dir: str) -> Dict[str, Any]:
    """Analyze if the entity relation graph is scale-free and output a CCDF plot.

    This fits a power-law to the degree distribution tail using a linear fit on
    log10-CCDF vs log10(k), estimates the power-law exponent gamma, computes an R^2
    goodness-of-fit, saves a single CCDF plot, and returns a simple decision.

    Returns a dict with: is_scale_free, gamma, r2, k_min, n_tail, plot_path.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        degrees = np.array([d for _, d in entity_graph.degree() if d > 0], dtype=np.float64)
        if degrees.size < 5:
            return {
                "is_scale_free": False,
                "gamma": float("nan"),
                "r2": float("nan"),
                "k_min": 0,
                "n_tail": int(degrees.size),
                "plot_path": None,
            }

        k_min = int(max(1, np.floor(np.percentile(degrees, 80))))
        tail = degrees[degrees >= k_min]
        if tail.size < 5:
            return {
                "is_scale_free": False,
                "gamma": float("nan"),
                "r2": float("nan"),
                "k_min": k_min,
                "n_tail": int(tail.size),
                "plot_path": None,
            }

        vals, counts = np.unique(tail.astype(int), return_counts=True)
        ccdf_counts = np.cumsum(counts[::-1])[::-1]
        ccdf = ccdf_counts / ccdf_counts[0]

        x = np.log10(vals.astype(np.float64))
        y = np.log10(ccdf.astype(np.float64))

        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept

        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.inf)

        gamma = 1.0 - slope  # CCDF slope ≈ -(gamma - 1) => gamma = 1 - slope

        plausible_gamma = 2.0 <= gamma <= 3.5
        good_fit = r2 >= 0.9
        is_scale_free = bool(plausible_gamma and good_fit)

        plt.figure(figsize=(6, 4))
        plt.scatter(vals, ccdf, s=12, alpha=0.7, label="Empirical CCDF")
        k_line = np.linspace(vals.min(), vals.max(), 200)
        ccdf_fit = (10 ** intercept) * (k_line ** slope)
        plt.plot(k_line, ccdf_fit, "r-", lw=2, label=f"Fit (gamma={gamma:.2f}, R^2={r2:.2f})")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Degree k (log)")
        plt.ylabel("P(K ≥ k) (log)")
        plt.title("Entity Graph Degree CCDF and Power-law Fit")
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "entity_graph_scale_free_ccdf.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "is_scale_free": is_scale_free,
            "gamma": float(gamma),
            "r2": float(r2),
            "k_min": int(k_min),
            "n_tail": int(tail.size),
            "plot_path": plot_path,
        }
    except Exception as e:
        logger.warning(f"Scale-free analysis failed: {e}")
        return {
            "is_scale_free": False,
            "gamma": float("nan"),
            "r2": float("nan"),
            "k_min": 0,
            "n_tail": 0,
            "plot_path": None,
            "error": str(e),
        }

def save_top_k_entity_nodes_plot(entity_graph: nx.Graph, output_dir: str, k: int = 10) -> Optional[str]:
    """Save a horizontal bar plot of the top-k ENTITY_CONCEPT nodes by degree in the entity graph.

    Returns the plot path or None on failure.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Consider only ENTITY_CONCEPT nodes present in the entity graph
        entity_nodes = [n for n, d in entity_graph.nodes(data=True) if d.get('node_type') == 'ENTITY_CONCEPT']
        if not entity_nodes:
            return None
        degrees = dict(entity_graph.degree(entity_nodes))
        if not degrees:
            return None
        # Top-k by degree
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max(1, k)]
        labels = []
        values = []
        for node_id, deg in top_nodes:
            node_name = entity_graph.nodes[node_id].get('name') or str(node_id)
            # Keep labels readable
            if isinstance(node_name, str) and len(node_name) > 43:
                node_name = node_name[:40] + '...'
            labels.append(node_name)
            values.append(deg)
        if not values:
            return None
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(values))
        plt.barh(y_pos, values, color='steelblue')
        plt.yticks(y_pos, labels)
        plt.gca().invert_yaxis()
        plt.xlabel('Degree (entity_relation)')
        plt.title(f'Top {len(values)} entities by degree')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'top10_entity_nodes.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        return plot_path
    except Exception as e:
        logger.warning(f"Top-k entity nodes plot generation failed: {e}")
        return None

async def _save_results(
    enhanced_graph: nx.DiGraph,
    entity_graph: nx.Graph,
    communities: Dict[str, int],
    subcommunities: Dict[str, int],
    statistics: Dict[str, Any],
    processing_time: float,
    output_dir: str,
    embeddings: Dict[str, np.ndarray] = None,
    centrality_results: Dict[str, Any] = None,
    community_quality: CommunityQualityMetrics = None
):
    """Save all results to files"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save enhanced graph (compatible with topic_summarizer.py)
    graph_file = os.path.join(output_dir, 'semantic_graph_with_communities.json')
    graph_data = filter_embedding_attributes_from_graph(enhanced_graph)
    
    with open(graph_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False, default=json_serialize_dates)
    
    logger.info(f"Enhanced graph saved to {graph_file}")
    
    # Save entity graph
    entity_graph_file = os.path.join(output_dir, 'semantic_entity_graph.json')
    entity_graph_data = filter_embedding_attributes_from_graph(entity_graph)
    
    with open(entity_graph_file, 'w', encoding='utf-8') as f:
        json.dump(entity_graph_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Entity graph saved to {entity_graph_file}")
    
    # Save community assignments
    community_file = os.path.join(output_dir, 'semantic_communities.json')
    community_data = {
        "communities": communities,
        "subcommunities": subcommunities,
        "processing_time": processing_time,
        "methodology": "semantic_with_communities"
    }
    
    with open(community_file, 'w', encoding='utf-8') as f:
        json.dump(community_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Communities saved to {community_file}")
    
    # Save comprehensive statistics
    stats_file = os.path.join(output_dir, 'semantic_statistics.json')
    full_stats = {
        **statistics,
        "processing_time": processing_time,
        "output_files": {
            "enhanced_graph": graph_file,
            "entity_graph": entity_graph_file,
            "communities": community_file
        }
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(full_stats, f, indent=2, ensure_ascii=False, default=json_serialize_dates)
    
    logger.info(f"Statistics saved to {stats_file}")
    
    # Generate PyVis HTML visualization
    if PYVIS_AVAILABLE:
        viz_file = os.path.join(output_dir, 'semantic_graph_visualization.html')
        generate_pyvis_visualization(
            enhanced_graph, 
            viz_file,
            title="Semantic Knowledge Graph with Communities",
            communities=communities
        )
        full_stats["output_files"]["visualization"] = viz_file
    
    # RAG embeddings saving removed per user request

    # Save Top-10 entity nodes plot (entity-relation graph)
    try:
        top_plot = save_top_k_entity_nodes_plot(entity_graph, output_dir, k=10)
        if top_plot:
            full_stats["output_files"]["top_entities_plot"] = top_plot
    except Exception as _:
        pass
    
    # Add community summary comparison results to output files
    comparison_file = os.path.join(output_dir, 'community_summary_comparison.json')
    if os.path.exists(comparison_file):
        full_stats["output_files"]["summary_comparison"] = comparison_file
    
    # Add heatmap to output files if it exists
    heatmap_file = os.path.join(output_dir, 'community_summary_similarity_heatmap.png')
    if os.path.exists(heatmap_file):
        full_stats["output_files"]["similarity_heatmap"] = heatmap_file
    
    # Add histogram to output files if it exists
    histogram_file = os.path.join(output_dir, 'community_summary_similarity_histogram.png')
    if os.path.exists(histogram_file):
        full_stats["output_files"]["similarity_histogram"] = histogram_file
    
    # Add embeddings scatterplot to output files if it exists
    scatter_file = os.path.join(output_dir, 'community_summary_embeddings_scatter.png')
    if os.path.exists(scatter_file):
        full_stats["output_files"]["summary_embeddings_scatter"] = scatter_file
    
    # Generate executive summary markdown if centrality and community quality data are available
    if centrality_results is not None and community_quality is not None:
        try:
            executive_summary_file = generate_executive_summary_markdown(
                enhanced_graph, entity_graph, communities, subcommunities,
                centrality_results, community_quality, statistics, processing_time, output_dir
            )
            full_stats["output_files"]["executive_summary"] = executive_summary_file
            logger.info(f"Executive summary generated: {executive_summary_file}")
        except Exception as e:
            logger.warning(f"Executive summary generation failed: {e}")
    
    # Update stats file with new paths
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(full_stats, f, indent=2, ensure_ascii=False, default=json_serialize_dates)

# =============================================================================
# INTEGRATED SEMANTIC KG WITH TOPIC SUMMARIZATION
# =============================================================================

async def build_semantic_kg_with_topics(
    input_dir: str = None,
    output_dir: str = None,
    save_files: bool = True,
    config_overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Build semantic knowledge graph with integrated community detection and topic summarization.
    
    This function combines the semantic KG building with topic summarization in one step,
    replacing the need to call both functions separately.
    
    Args:
        input_dir: Override input directory
        output_dir: Override output directory  
        save_files: Whether to save output files
        config_overrides: Dictionary of configuration overrides (e.g., {'speech_limit': 100})
        
    Returns:
        Dictionary containing both semantic KG results and topic summarization results
    """
    
    logger.info("Starting Integrated Semantic Knowledge Graph with Topic Summarization...")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Build semantic KG with communities
        logger.info("Phase 1: Building semantic knowledge graph with communities...")
        semantic_results = await build_semantic_kg_with_communities(
            input_dir=input_dir,
            output_dir=output_dir,
            save_files=save_files,
            config_overrides=config_overrides
        )
        
        logger.info(f"Semantic KG completed: {semantic_results.base_graph.number_of_nodes()} nodes, {semantic_results.base_graph.number_of_edges()} edges")
        logger.info(f"Communities: {len(set(semantic_results.communities.values()))}, Subcommunities: {len(set(semantic_results.subcommunities.values()))}")
        
        # Step 2: Run topic summarization on the graph
        logger.info("Phase 2: Running topic summarization...")
        
        # Use the enhanced graph for topic summarization
        graph_for_topics = semantic_results.enhanced_graph
        
        # Create a temporary file path for the graph (required by topic summarizer)
        if output_dir is None:
            output_dir = create_output_directory()
        
        temp_graph_file = os.path.join(output_dir, 'temp_semantic_graph.json')
        final_graph_file = os.path.join(output_dir, 'semantic_graph_with_topics.json')
        
        # Save the enhanced graph temporarily for topic summarization (filter out embedding attributes)
        # Define embedding-related attribute names to filter out
        embedding_attributes = {
            'kge_embedding', 'embedding', 'embeddings', 'vector', 'vectors',
            'embedding_vector', 'node_embedding', 'entity_embedding',
            'semantic_embedding', 'graph_embedding', 'feature_vector'
        }
        
        def filter_node_data(node_data):
            """Filter out embedding attributes from node data"""
            filtered_data = {}
            for key, value in node_data.items():
                if key.lower() not in embedding_attributes and not key.lower().endswith('_embedding'):
                    filtered_data[key] = value
            return filtered_data
        
        graph_data = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [
                {
                    "id": str(node_id),
                    "data": filter_node_data(node_data)
                }
                for node_id, node_data in graph_for_topics.nodes(data=True)
            ],
            "links": [
                {
                    "source": str(source),
                    "target": str(target),
                    "data": edge_data
                }
                for source, target, edge_data in graph_for_topics.edges(data=True)
            ]
        }
        
        with open(temp_graph_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False, default=json_serialize_dates)
        
        # Run topic summarization
        topic_results = await run_topic_summarization_on_graph(
            graph=graph_for_topics,
            graph_file_path=temp_graph_file,
            output_file_path=final_graph_file
        )
        
        # Clean up temporary file
        if os.path.exists(temp_graph_file):
            os.remove(temp_graph_file)
        
        logger.info(f"Topic summarization completed: {topic_results['topics_updated']} topics, {topic_results['subtopics_updated']} subtopics")
        
        total_processing_time = (datetime.now() - start_time).total_seconds()
        
        # Combine results
        combined_results = {
            "success": True,
            "semantic_kg_results": semantic_results,
            "topic_summarization_results": topic_results,
            "total_processing_time": total_processing_time,
            "output_files": {
                "graph_file": final_graph_file,
                "similarity_file": topic_results.get("similarity_file_path"),
                "visualization_file": semantic_results.statistics.get("output_files", {}).get("visualization"),
                "embeddings_file": semantic_results.statistics.get("output_files", {}).get("embeddings"),
                "stats_file": semantic_results.statistics.get("output_files", {}).get("stats")
            }
        }
        
        logger.info(f"✅ Integrated pipeline completed in {total_processing_time:.2f} seconds")
        logger.info(f"📊 Final graph with topics saved to: {final_graph_file}")
        
        return combined_results
        
    except Exception as e:
        logger.error(f"Integrated pipeline failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "total_processing_time": (datetime.now() - start_time).total_seconds()
        }

# =============================================================================
# CLAIM GENERATION FOR CHUNKS AND COMMUNITY SUMMARY GENERATION (minimal, additive)
# =============================================================================

def _claims_prompt(text: str, entities: List[str]) -> str:
    # Ensure deterministic, explicit entity scoping for claim generation
    entities_list = ", ".join(sorted(dict.fromkeys([e for e in entities if isinstance(e, str) and e.strip()])))
    return (
        "You are an experienced text analyst who identifies salient claims in text.\n\n"
        "TASK\nAnalyze the text below and extract claims that directly involve the specified entities."
        " Only write claims that explicitly mention at least one of these entities by name.\n\n"
        f"FOCUS ENTITIES\n{entities_list}\n\n"
        "FOCUS AREAS\n- Factual statements and objective information about the entities\n"
        "- Opinions/evaluations directed at the entities\n- Arguments/justifications referencing the entities\n- Key statements and conclusions tied to the entities\n\n"
        "TEXT TO ANALYZE\n---\n" + text + "\n---\n\n"
        "REQUIREMENTS\n- Each claim must reference at least one listed entity by name\n"
        "- Concise bullet list; each bullet is a single complete sentence\n"
        "- Avoid generic statements not grounded in the entities\n\n"
        "ANSWER\nList the claims:"
    )

def _is_invalid_claims(text: str) -> bool:
    if not text or not str(text).strip():
        return True
    stripped = str(text).strip()
    invalid_headers = {"Állítások:", "Írd ide az állításokat:", "Összefoglaló:", "Írd ide az összefoglalót:"}
    return stripped in invalid_headers

def generate_claims_for_chunks(graph: nx.DiGraph, batch_log_every: int = 25) -> Dict[str, int]:
    """Generate entity-focused claim bullets for CHUNK nodes using existing llm.

    Claims are constrained to entities extracted for the chunk (via HAS_ENTITY edges),
    falling back to the chunk's initial entity candidates if enrichment hasn't run yet.
    """
    if not llm:
        logger.warning("LLM not available; skipping chunk claims generation")
        return {"total": 0, "valid": 0, "invalid": 0}
    total = 0
    valid = 0
    invalid = 0
    processed = 0
    from langchain_core.prompts import ChatPromptTemplate

    def _entities_for_chunk(cid: str) -> List[str]:
        names: List[str] = []
        try:
            # Prefer canonical entities connected via HAS_ENTITY
            for pred in graph.predecessors(cid):
                edge = graph.get_edge_data(pred, cid) or {}
                # We expect CHUNK -> ENTITY edges (HAS_ENTITY) to be from chunk to entity;
                # but if predecessors are used, also check successor direction below
            # Successors: CHUNK -> ENTITY
            for succ in graph.successors(cid):
                edge = graph.get_edge_data(cid, succ) or {}
                if edge.get('label') == 'HAS_ENTITY' and graph.nodes.get(succ, {}).get('node_type') == 'ENTITY_CONCEPT':
                    nm = graph.nodes[succ].get('name') or str(succ)
                    names.append(str(nm))
        except Exception:
            pass
        if not names:
            # Fallback to initial_entities captured during chunking
            initial = graph.nodes.get(cid, {}).get('initial_entities') or []
            names.extend([str(x) for x in initial if isinstance(x, str)])
        # Deduplicate while preserving order
        dedup: List[str] = []
        seen = set()
        for n in names:
            if n not in seen and n.strip():
                seen.add(n)
                dedup.append(n)
        return dedup

    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('node_type') != 'CHUNK':
            continue
        # Prefer 'text' if present, else join 'sentences'
        chunk_text = node_data.get('text')
        if not chunk_text:
            sentences = node_data.get('sentences', [])
            if isinstance(sentences, list):
                chunk_text = " ".join(sentences)
            else:
                chunk_text = str(sentences) if sentences else ""
        if not chunk_text or not chunk_text.strip():
            continue
        # Determine scoped entities for this chunk
        scoped_entities = _entities_for_chunk(node_id)
        if not scoped_entities:
            # If no entities, skip generating claims for this chunk
            graph.nodes[node_id]['claims'] = ""
            graph.nodes[node_id]['claims_valid'] = False
            graph.nodes[node_id]['claims_length'] = 0
            graph.nodes[node_id]['claims_error'] = 'No scoped entities available for claim generation'
            continue

        total += 1

        try:
            prompt = ChatPromptTemplate.from_template("{prompt}")
            chain = prompt | llm
            response = chain.invoke({"prompt": _claims_prompt(chunk_text, scoped_entities)})
            content = response.content if hasattr(response, 'content') else str(response)
            is_valid = not _is_invalid_claims(content)
            graph.nodes[node_id]['claims'] = content
            graph.nodes[node_id]['claims_valid'] = bool(is_valid)
            graph.nodes[node_id]['claims_length'] = len(content) if content else 0
            graph.nodes[node_id]['claims_scope_entities'] = scoped_entities
            if is_valid:
                valid += 1
            else:
                invalid += 1
                graph.nodes[node_id]['claims_error'] = 'Invalid claims content'
        except Exception as e:
            graph.nodes[node_id]['claims'] = ""
            graph.nodes[node_id]['claims_valid'] = False
            graph.nodes[node_id]['claims_length'] = 0
            graph.nodes[node_id]['claims_error'] = f"Error generating claims: {str(e)}"
            invalid += 1

        processed += 1
        if processed % max(1, batch_log_every) == 0:
            logger.info(f"Generated claims for {processed} chunks (valid={valid}, invalid={invalid})")

    logger.info(f"Chunk claims generation complete: total={total}, valid={valid}, invalid={invalid}")
    return {"total": total, "valid": valid, "invalid": invalid}

async def generate_claims_for_chunks_async(graph: nx.DiGraph, batch_log_every: int = 25, max_concurrent: int = None) -> Dict[str, int]:
    """Async variant of claim generation, mirroring generate_claims_for_chunks but using ainvoke with concurrency."""
    if not llm:
        logger.warning("LLM not available; skipping chunk claims generation (async)")
        return {"total": 0, "valid": 0, "invalid": 0}
    from langchain_core.prompts import ChatPromptTemplate
    total = 0
    valid = 0
    invalid = 0
    processed = 0
    max_conc = max_concurrent if isinstance(max_concurrent, int) and max_concurrent > 0 else MAX_CONCURRENT_SUMMARIES

    def _entities_for_chunk(cid: str) -> List[str]:
        names: List[str] = []
        try:
            for succ in graph.successors(cid):
                edge = graph.get_edge_data(cid, succ) or {}
                if edge.get('label') == 'HAS_ENTITY' and graph.nodes.get(succ, {}).get('node_type') == 'ENTITY_CONCEPT':
                    nm = graph.nodes[succ].get('name') or str(succ)
                    names.append(str(nm))
        except Exception:
            pass
        if not names:
            initial = graph.nodes.get(cid, {}).get('initial_entities') or []
            names.extend([str(x) for x in initial if isinstance(x, str)])
        dedup: List[str] = []
        seen = set()
        for n in names:
            if n not in seen and n.strip():
                seen.add(n)
                dedup.append(n)
        return dedup

    async def _process_chunk(cid: str, semaphore: asyncio.Semaphore) -> Tuple[bool, str]:
        nonlocal total, valid, invalid, processed
        node_data = graph.nodes.get(cid, {})
        chunk_text = node_data.get('text')
        if not chunk_text:
            sentences = node_data.get('sentences', [])
            if isinstance(sentences, list):
                chunk_text = " ".join(sentences)
            else:
                chunk_text = str(sentences) if sentences else ""
        if not chunk_text or not chunk_text.strip():
            return (False, cid)
        scoped_entities = _entities_for_chunk(cid)
        if not scoped_entities:
            graph.nodes[cid]['claims'] = ""
            graph.nodes[cid]['claims_valid'] = False
            graph.nodes[cid]['claims_length'] = 0
            graph.nodes[cid]['claims_error'] = 'No scoped entities available for claim generation'
            return (False, cid)

        total += 1
        async with semaphore:
            try:
                prompt = ChatPromptTemplate.from_template("{prompt}")
                chain = prompt | llm
                response = await chain.ainvoke({"prompt": _claims_prompt(chunk_text, scoped_entities)})
                content = response.content if hasattr(response, 'content') else str(response)
                is_valid = not _is_invalid_claims(content)
                graph.nodes[cid]['claims'] = content
                graph.nodes[cid]['claims_valid'] = bool(is_valid)
                graph.nodes[cid]['claims_length'] = len(content) if content else 0
                graph.nodes[cid]['claims_scope_entities'] = scoped_entities
                if is_valid:
                    valid += 1
                else:
                    invalid += 1
                    graph.nodes[cid]['claims_error'] = 'Invalid claims content'
            except Exception as e:
                graph.nodes[cid]['claims'] = ""
                graph.nodes[cid]['claims_valid'] = False
                graph.nodes[cid]['claims_length'] = 0
                graph.nodes[cid]['claims_error'] = f"Error generating claims: {str(e)}"
                invalid += 1
            finally:
                processed += 1
                if processed % max(1, batch_log_every) == 0:
                    logger.info(f"Generated claims (async) for {processed} chunks (valid={valid}, invalid={invalid})")
        return (True, cid)

    semaphore = asyncio.Semaphore(max_conc)
    tasks = []
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('node_type') != 'CHUNK':
            continue
        tasks.append(_process_chunk(node_id, semaphore))
    if tasks:
        await asyncio.gather(*tasks)
    logger.info(f"Chunk claims generation complete (async): total={total}, valid={valid}, invalid={invalid}")
    return {"total": total, "valid": valid, "invalid": invalid}

def _collect_chunk_claims_for_subcommunity(graph: nx.DiGraph, subcomm_node: str) -> List[str]:
    claims: List[str] = []
    # Entities -> subcommunity via IN_COMMUNITY
    for pred in graph.predecessors(subcomm_node):
        edge = graph.get_edge_data(pred, subcomm_node) or {}
        if edge.get('label') != 'IN_COMMUNITY':
            continue
        if graph.nodes[pred].get('node_type') != 'ENTITY_CONCEPT':
            continue
        # Chunks -> entity via HAS_ENTITY
        for chunk_pred in graph.predecessors(pred):
            e = graph.get_edge_data(chunk_pred, pred) or {}
            if e.get('label') != 'HAS_ENTITY':
                continue
            if graph.nodes[chunk_pred].get('node_type') != 'CHUNK':
                continue
            cl = graph.nodes[chunk_pred].get('claims', '')
            if cl and str(graph.nodes[chunk_pred].get('claims_valid', True)):
                claims.append(cl.strip())
    return claims

def _collect_chunk_claims_for_parent(graph: nx.DiGraph, comm_node: str) -> List[str]:
    claims: List[str] = []
    # Prefer subcommunity summaries if available
    sub_summaries = []
    for pred in graph.predecessors(comm_node):
        edge = graph.get_edge_data(pred, comm_node) or {}
        if edge.get('label') == 'PARENT_COMMUNITY' and graph.nodes[pred].get('node_type') == 'SUBCOMMUNITY':
            s = graph.nodes[pred].get('community_summary', '')
            if s and s.strip():
                sub_summaries.append(s.strip())
    if sub_summaries:
        return sub_summaries

    # Fallback: entities connected directly to community via IN_COMMUNITY
    for pred in graph.predecessors(comm_node):
        edge = graph.get_edge_data(pred, comm_node) or {}
        if edge.get('label') != 'IN_COMMUNITY':
            continue
        if graph.nodes[pred].get('node_type') != 'ENTITY_CONCEPT':
            continue
        for chunk_pred in graph.predecessors(pred):
            e = graph.get_edge_data(chunk_pred, pred) or {}
            if e.get('label') != 'HAS_ENTITY':
                continue
            if graph.nodes[chunk_pred].get('node_type') != 'CHUNK':
                continue
            cl = graph.nodes[chunk_pred].get('claims', '')
            if cl and str(graph.nodes[chunk_pred].get('claims_valid', True)):
                claims.append(cl.strip())
    return claims

def _summarize_claims_llm(claims: List[str], kind: str) -> str:
    if not llm:
        return ""
    all_content = "\n\n".join(claims)
    community_type = 'subtopic' if kind == 'sub' else 'topic'
    prompt_text = (
        "You are an experienced content analyst summarizing topics based on extracted claims.\n\n"
        f"TASK\nAnalyze all claims of the {community_type} below and produce a comprehensive summary.\n\n"
        "CLAIMS\n---\n" + all_content + "\n---\n\n"
        "REQUIREMENTS\n- Produce a detailed summary (max 200 words)\n- Capture the main themes and claims of the "
        f"{community_type}\n- Be specific and informative\n- Avoid the technical term \"community\"; prefer \"topic\" or \"theme\"\n- Return only the summary"
    )
    from langchain_core.prompts import ChatPromptTemplate
    chain = ChatPromptTemplate.from_template("{p}") | llm
    resp = chain.invoke({"p": prompt_text})
    return resp.content if hasattr(resp, 'content') else str(resp)

def _summarize_subsummaries_llm(summaries: List[str]) -> str:
    if not llm:
        return ""
    all_content = "\n\n".join(summaries)
    prompt_text = (
        "You are an experienced content analyst summarizing topics based on subcommunity summaries.\n\n"
        "TASK\nAnalyze the summaries of the subcommunities below and produce a comprehensive topic summary.\n\n"
        "SUBCOMMUNITY SUMMARIES\n---\n" + all_content + "\n---\n\n"
        "REQUIREMENTS\n- Produce a detailed summary (max 200 words)\n- Capture the main themes of the topic as reflected by its subcommunities\n- Be specific and informative\n- Avoid the technical term \"community\"; prefer \"topic\" or \"theme\"\n- Return only the summary"
    )
    from langchain_core.prompts import ChatPromptTemplate
    chain = ChatPromptTemplate.from_template("{p}") | llm
    resp = chain.invoke({"p": prompt_text})
    return resp.content if hasattr(resp, 'content') else str(resp)

def generate_community_and_subcommunity_summaries_from_claims(graph: nx.DiGraph) -> Dict[str, str]:
    """Populate community_summary on SUBCOMMUNITY_* first, then COMMUNITY_* nodes."""
    summaries: Dict[str, str] = {}
    # Subcommunities first
    sub_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'SUBCOMMUNITY']
    logger.info(f"Processing {len(sub_nodes)} subcommunities for summaries from claims...")
    for sub in sub_nodes:
        claims = _collect_chunk_claims_for_subcommunity(graph, sub)
        if not claims:
            continue
        try:
            summary = _summarize_claims_llm(claims, kind='sub')
            if summary and summary.strip():
                graph.nodes[sub]['community_summary'] = summary
                summaries[sub] = summary
        except Exception as e:
            graph.nodes[sub]['community_summary_error'] = str(e)

    # Parent communities
    comm_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'COMMUNITY']
    logger.info(f"Processing {len(comm_nodes)} parent communities for summaries...")
    for com in comm_nodes:
        collected = _collect_chunk_claims_for_parent(graph, com)
        if not collected:
            continue
        try:
            # If collected are sub-summaries (strings that likely came from subcommunities), use subsummary prompt
            use_subs = False
            # Heuristic: assume they are subsummaries if any predecessor subcommunity exists
            for pred in graph.predecessors(com):
                if graph.nodes[pred].get('node_type') == 'SUBCOMMUNITY':
                    use_subs = True
                    break
            summary = _summarize_subsummaries_llm(collected) if use_subs else _summarize_claims_llm(collected, kind='top')
            if summary and summary.strip():
                graph.nodes[com]['community_summary'] = summary
                summaries[com] = summary
        except Exception as e:
            graph.nodes[com]['community_summary_error'] = str(e)

    logger.info(f"Community summary generation complete: {len(summaries)} summaries written")
    return summaries

def generate_community_summary_comparison(
    graph: nx.DiGraph,
    output_dir: str,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Compare all community and subcommunity summaries using embeddings.
    
    Generates:
    1. Embeddings for all summaries
    2. Similarity matrix heatmap
    3. Average similarity scores
    4. Top 10 most similar topic pairs
    
    Args:
        graph: NetworkX graph with community summaries
        output_dir: Directory to save results
        embedding_model: Model for generating embeddings
        
    Returns:
        Dictionary with comparison results and statistics
    """
    logger.info("Starting community summary comparison...")
    
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("SentenceTransformers not available. Skipping summary comparison.")
        return {"error": "SentenceTransformers not available"}
    
    # Collect all community and subcommunity summaries
    summaries = {}
    summary_nodes = []
    
    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get('node_type')
        if node_type in ['COMMUNITY', 'SUBCOMMUNITY']:
            summary = node_data.get('community_summary', '')
            if summary and summary.strip():
                summaries[node_id] = summary.strip()
                summary_nodes.append(node_id)
    
    if len(summaries) < 2:
        logger.warning(f"Not enough summaries for comparison: {len(summaries)} found")
        return {"error": f"Not enough summaries for comparison: {len(summaries)} found"}
    
    logger.info(f"Found {len(summaries)} summaries for comparison")
    
    try:
        # Generate embeddings
        model = SentenceTransformer(embedding_model)
        summary_texts = [summaries[node_id] for node_id in summary_nodes]
        embeddings = model.encode(summary_texts)
        
        # Store embeddings as node attributes
        for i, node_id in enumerate(summary_nodes):
            graph.nodes[node_id]['summary_embedding'] = embeddings[i].tolist()
        
        # Compute similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Calculate statistics
        # Get upper triangle (excluding diagonal) for pairwise similarities
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        avg_similarity = np.mean(upper_triangle)
        max_similarity = np.max(upper_triangle)
        min_similarity = np.min(upper_triangle)
        
        # Find top 10 most similar pairs
        similar_pairs = []
        for i in range(len(summary_nodes)):
            for j in range(i + 1, len(summary_nodes)):
                similarity = similarity_matrix[i][j]
                similar_pairs.append({
                    'node1': summary_nodes[i],
                    'node2': summary_nodes[j],
                    'similarity': float(similarity),
                    'summary1': summaries[summary_nodes[i]][:100] + "..." if len(summaries[summary_nodes[i]]) > 100 else summaries[summary_nodes[i]],
                    'summary2': summaries[summary_nodes[j]][:100] + "..." if len(summaries[summary_nodes[j]]) > 100 else summaries[summary_nodes[j]]
                })
        
        # Sort by similarity (descending)
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        top_10_pairs = similar_pairs[:10]
        
        # Generate visualizations if available
        heatmap_path = None
        histogram_path = None
        scatter_path = None
        if VISUALIZATION_AVAILABLE:
            try:
                # Create figure with subplots
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(28, 8))
                
                # Plot 1: Histogram of similarities (excluding diagonal)
                ax1.hist(upper_triangle, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.axvline(avg_similarity, color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {avg_similarity:.3f}')
                ax1.axvline(max_similarity, color='orange', linestyle='--', linewidth=2, 
                           label=f'Max: {max_similarity:.3f}')
                ax1.axvline(min_similarity, color='green', linestyle='--', linewidth=2, 
                           label=f'Min: {min_similarity:.3f}')
                ax1.set_xlabel('Cosine Similarity')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Distribution of Community Summary Similarities')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Clean heatmap without annotations
                im = ax2.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto')
                ax2.set_title('Community Summary Similarity Matrix')
                ax2.set_xlabel('Community/Subcommunity Index')
                ax2.set_ylabel('Community/Subcommunity Index')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax2)
                cbar.set_label('Cosine Similarity')
                
                # Plot 3: 2D scatterplot of summary embeddings (PCA)
                try:
                    from sklearn.decomposition import PCA
                    import matplotlib.patches as mpatches
                    import matplotlib.cm as cm
                    import matplotlib.colors as mcolors
                    pca = PCA(n_components=2)
                    coords_2d = pca.fit_transform(embeddings)
                    
                    # Determine parent community id for each node
                    parent_ids = []
                    for nid in summary_nodes:
                        if isinstance(nid, str) and nid.startswith('SUBCOMMUNITY_'):
                            # Format: SUBCOMMUNITY_{parent}_{local}
                            parts = nid.split('_')
                            parent_ids.append(parts[1])
                        elif isinstance(nid, str) and nid.startswith('COMMUNITY_'):
                            parent_ids.append(nid.split('_')[1])
                        else:
                            parent_ids.append('UNKNOWN')
                    
                    # Map each unique parent id to a color using a categorical colormap
                    unique_parents = sorted(list(dict.fromkeys(parent_ids)))
                    num_parents = len(unique_parents)
                    # Choose a colormap with sufficient categories
                    base_cmap = cm.get_cmap('tab20', max(2, min(20, num_parents)))
                    # If more than 20, fall back to hsv for broader spread
                    if num_parents > 20:
                        base_cmap = cm.get_cmap('hsv', num_parents)
                    parent_to_color: Dict[str, Any] = {}
                    for idx, pid in enumerate(unique_parents):
                        parent_to_color[pid] = base_cmap(idx % base_cmap.N)
                    
                    colors = [parent_to_color[pid] for pid in parent_ids]
                    sc = ax3.scatter(coords_2d[:, 0], coords_2d[:, 1], c=colors, alpha=0.85, s=30, edgecolors='none')
                    ax3.set_title('Summary Embeddings (PCA) colored by Parent Community')
                    ax3.set_xlabel('PC1')
                    ax3.set_ylabel('PC2')
                    ax3.grid(True, alpha=0.2)
                    
                    # Build a compact legend: cap at 20 entries to avoid clutter
                    legend_handles = []
                    max_legend = 20
                    for idx, pid in enumerate(unique_parents[:max_legend]):
                        legend_handles.append(mpatches.Patch(color=parent_to_color[pid], label=f'COMMUNITY_{pid}'))
                    if legend_handles:
                        ax3.legend(handles=legend_handles, title='Parent Community', fontsize=8, title_fontsize=9, loc='best', framealpha=0.8)
                except Exception as _:
                    # If PCA or plotting fails, leave ax3 blank but keep figure generation
                    ax3.set_visible(False)
                
                plt.tight_layout()
                
                # Save plots
                histogram_path = os.path.join(output_dir, 'community_summary_similarity_histogram.png')
                heatmap_path = os.path.join(output_dir, 'community_summary_similarity_heatmap.png')
                scatter_path = os.path.join(output_dir, 'community_summary_embeddings_scatter.png')
                
                plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Similarity histogram saved to {histogram_path}")
                logger.info(f"Similarity heatmap saved to {heatmap_path}")
                logger.info(f"Embeddings scatter saved to {scatter_path}")
            except Exception as e:
                logger.warning(f"Failed to generate visualizations: {e}")
                heatmap_path = None
                histogram_path = None
                scatter_path = None
        
        # Save detailed results
        results = {
            'total_summaries': len(summaries),
            'summary_nodes': summary_nodes,
            'similarity_matrix': similarity_matrix.tolist(),
            'statistics': {
                'average_similarity': float(avg_similarity),
                'max_similarity': float(max_similarity),
                'min_similarity': float(min_similarity),
                'std_similarity': float(np.std(upper_triangle))
            },
            'top_10_similar_pairs': top_10_pairs,
            'heatmap_path': heatmap_path,
            'histogram_path': histogram_path,
            'scatter_path': scatter_path,
            'embedding_model': embedding_model
        }
        
        # Save results to JSON
        results_file = os.path.join(output_dir, 'community_summary_comparison.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Community summary comparison results saved to {results_file}")
        logger.info(f"Average similarity: {avg_similarity:.3f}")
        logger.info(f"Most similar pair: {top_10_pairs[0]['node1']} <-> {top_10_pairs[0]['node2']} ({top_10_pairs[0]['similarity']:.3f})")
        
        return results
        
    except Exception as e:
        logger.error(f"Community summary comparison failed: {e}")
        return {"error": str(e)}

def save_distinct_low_similarity_topics(
    graph: nx.DiGraph,
    comparison_results: Dict[str, Any],
    output_dir: str,
    threshold_percentile: int = 10
) -> Optional[str]:
    """
    Compute and persist the most distinct communities/subcommunities by average cosine similarity.
    Uses in-memory comparison_results produced by generate_community_summary_comparison.

    Returns the output file path or None if unavailable.
    """
    try:
        summary_nodes = comparison_results.get('summary_nodes') or []
        sim_mat = comparison_results.get('similarity_matrix')
        if not summary_nodes or not sim_mat:
            return None

        import numpy as _np
        M = _np.asarray(sim_mat, dtype=float)
        n = M.shape[0]
        if n < 2:
            return None

        # Average similarity per community (exclude diagonal self-similarity of 1.0)
        avg_sims = (M.sum(axis=1) - 1.0) / max(1, (n - 1))
        thr = _np.percentile(avg_sims, threshold_percentile)

        results: List[Dict[str, Any]] = []
        for i, node_id in enumerate(summary_nodes):
            if avg_sims[i] <= thr:
                summary = ""
                if graph.has_node(node_id):
                    summary = (graph.nodes[node_id].get('community_summary') or
                               graph.nodes[node_id].get('summary') or "")
                results.append({
                    "node_id": node_id,
                    "average_similarity": float(avg_sims[i]),
                    "summary": summary if summary else "Summary not found."
                })

        # Sort most distinct first (lowest average similarity)
        results.sort(key=lambda x: x["average_similarity"])

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "distinct_topic_summaries.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return out_path
    except Exception:
        return None

def test_executive_summary_generation():
    """Test function to verify executive summary generation works correctly"""
    logger.info("Testing executive summary generation...")
    
    # Create minimal test data
    base_graph = nx.DiGraph()
    entity_graph = nx.Graph()
    
    # Add some test nodes
    base_graph.add_node("doc1", node_type="DOCUMENT", name="Test Document")
    base_graph.add_node("speaker1", node_type="SPEAKER", name="Test Speaker")
    base_graph.add_node("speech1", node_type="SPEECH", name="Test Speech")
    base_graph.add_node("chunk1", node_type="CHUNK", name="Test Chunk", sentences=["Test sentence 1", "Test sentence 2"])
    base_graph.add_node("entity1", node_type="ENTITY_CONCEPT", name="Test Entity", entity_type="PERSON")
    base_graph.add_node("entity2", node_type="ENTITY_CONCEPT", name="Another Entity", entity_type="ORGANIZATION")
    
    # Add some edges
    base_graph.add_edge("chunk1", "entity1", label="HAS_ENTITY")
    base_graph.add_edge("chunk1", "entity2", label="HAS_ENTITY")
    
    # Add to entity graph
    entity_graph.add_node("entity1")
    entity_graph.add_node("entity2")
    entity_graph.add_edge("entity1", "entity2")
    
    # Create test communities
    communities = {"entity1": 0, "entity2": 1}
    subcommunities = {"entity1": (0, 0), "entity2": (1, 0)}
    
    # Create test centrality results
    centrality_results = {
        'nodes_processed': 2,
        'centrality_measures': {
            'degree': {"entity1": 0.5, "entity2": 0.5},
            'pagerank': {"entity1": 0.6, "entity2": 0.4}
        },
        'statistics': {
            'degree': {'mean': 0.5, 'std': 0.0, 'min': 0.5, 'max': 0.5, 'median': 0.5, 'q25': 0.5, 'q75': 0.5},
            'pagerank': {'mean': 0.5, 'std': 0.1, 'min': 0.4, 'max': 0.6, 'median': 0.5, 'q25': 0.45, 'q75': 0.55}
        },
        'top_nodes': {
            'degree': [("entity1", 0.5), ("entity2", 0.5)],
            'pagerank': [("entity1", 0.6), ("entity2", 0.4)]
        }
    }
    
    # Create test community quality
    community_quality = CommunityQualityMetrics(
        modularity=0.3,
        size_variance=0.0,
        avg_size=1.0,
        min_size=1,
        max_size=1,
        size_ratio=1.0,
        text_length_variance=0.0,
        optimal_for_summarization=True
    )
    
    # Create test statistics
    statistics = {
        'base_graph_stats': {
            'total_nodes': 6,
            'total_edges': 2,
            'documents': 1,
            'speakers': 1,
            'speeches': 1,
            'chunks': 1,
            'entities_concepts': 2
        },
        'entity_graph_stats': {
            'entity_nodes': 2,
            'entity_edges': 1,
            'density': 1.0,
            'avg_degree': 1.0
        },
        'community_stats': {
            'num_communities': 2,
            'num_subcommunities': 2,
            'community_sizes': {'min': 1, 'max': 1, 'avg': 1.0, 'distribution': {0: 1, 1: 1}},
            'subcommunity_sizes': {'min': 1, 'max': 1, 'avg': 1.0, 'distribution': {'0_0': 1, '1_0': 1}}
        }
    }
    
    # Test executive summary generation
    output_dir = "/tmp/test_executive_summary"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        summary_file = generate_executive_summary_markdown(
            base_graph, entity_graph, communities, subcommunities,
            centrality_results, community_quality, statistics, 10.5, output_dir
        )
        
        # Verify file was created
        if os.path.exists(summary_file):
            logger.info(f"✅ Executive summary test passed: {summary_file}")
            
            # Read and verify content
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for key sections
            required_sections = [
                "# Knowledge Graph Executive Summary",
                "## Graph Structure Metrics",
                "## Centrality Measures",
                "## Community Structure Analysis",
                "## Processing Performance",
                "## Key Insights",
                "## Recommendations"
            ]
            
            missing_sections = [section for section in required_sections if section not in content]
            if missing_sections:
                logger.warning(f"Missing sections in executive summary: {missing_sections}")
            else:
                logger.info("✅ All required sections present in executive summary")
                
            # Check for key metrics
            required_metrics = [
                "Total Nodes: 6",
                "Total Edges: 2",
                "Documents: 1",
                "Entities/Concepts: 2",
                "Number of Communities: 2",
                "Modularity: 0.3000",
                "Processing Time: 10.50 seconds"
            ]
            
            missing_metrics = [metric for metric in required_metrics if metric not in content]
            if missing_metrics:
                logger.warning(f"Missing metrics in executive summary: {missing_metrics}")
            else:
                logger.info("✅ All required metrics present in executive summary")
                
            return True
        else:
            logger.error("❌ Executive summary file was not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Executive summary test failed: {e}")
        return False

async def main():
    """Main function for testing the integrated semantic KG with communities"""
    logger.info("Testing Integrated Semantic Knowledge Graph with Community Detection...")
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Run the integrated pipeline
    results = await build_semantic_kg_with_communities(
        output_dir=output_dir,
        save_files=True
    )
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATED SEMANTIC KNOWLEDGE GRAPH WITH COMMUNITIES - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Processing Time: {results.processing_time:.2f} seconds")
    logger.info(f"Base Graph: {results.base_graph.number_of_nodes()} nodes, {results.base_graph.number_of_edges()} edges")
    logger.info(f"Entity Graph: {results.entity_graph.number_of_nodes()} nodes, {results.entity_graph.number_of_edges()} edges")
    logger.info(f"Communities: {len(set(results.communities.values()))}")
    logger.info(f"Subcommunities: {len(set(results.subcommunities.values()))}")
    logger.info(f"Community Quality - Modularity: {results.community_quality.modularity:.3f}")
    logger.info(f"Community Quality - Optimal for Summarization: {results.community_quality.optimal_for_summarization}")
    
    # Centrality analysis results
    centrality_analysis = results.statistics.get('centrality_analysis', {})
    if centrality_analysis and not centrality_analysis.get('error'):
        logger.info(f"Centrality Analysis: {centrality_analysis['nodes_processed']} entity-relation nodes processed")
        logger.info(f"Centrality Measures: {len(centrality_analysis['centrality_measures'])} measures calculated")
        if centrality_analysis.get('top_nodes'):
            logger.info("Top entities by centrality:")
            for measure_name, top_nodes in centrality_analysis['top_nodes'].items():
                if top_nodes:
                    top_entity = top_nodes[0]
                    logger.info(f"  {measure_name}: {top_entity[0]} (score: {top_entity[1]:.4f})")
    else:
        logger.warning(f"Centrality analysis failed: {centrality_analysis.get('error', 'Unknown error')}")
    
    # Scale-free analysis and plot
    sf = analyze_scale_free_entity_graph(results.entity_graph, output_dir)
    logger.info(
        f"Scale-free: {sf['is_scale_free']} | gamma={sf['gamma']:.2f} | R^2={sf['r2']:.3f} | k_min={sf['k_min']} | tail_n={sf['n_tail']}"
    )
    if sf.get("plot_path"):
        logger.info(f"Scale-free CCDF plot saved to {sf['plot_path']}")
    
    # Print detailed statistics
    stats = results.statistics
    logger.info("\nDetailed Statistics:")
    logger.info(f"  Documents: {stats['base_graph_stats']['documents']}")
    logger.info(f"  Speakers: {stats['base_graph_stats']['speakers']}")
    logger.info(f"  Speeches: {stats['base_graph_stats']['speeches']}")
    logger.info(f"  Chunks: {stats['base_graph_stats']['chunks']}")
    logger.info(f"  Entities/Concepts: {stats['base_graph_stats']['entities_concepts']}")
    
    if stats['community_stats']['community_sizes']:
        comm_sizes = stats['community_stats']['community_sizes']
        logger.info(f"  Community Sizes - Min: {comm_sizes['min']}, Max: {comm_sizes['max']}, Avg: {comm_sizes['avg']:.1f}")
    
    logger.info("=" * 80)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
