"""
Integration tests for pipeline stage dependency resolution.
"""

import pytest
from src.kg.config import Config
from src.kg.pipeline.stages import registry, PipelineStage


class TestDependencyResolution:
    """Test automatic dependency resolution."""
    
    def test_community_detection_enables_dependencies(self):
        """Test that enabling community_detection auto-enables its dependencies."""
        config = Config(
            pipeline={
                "stages": {
                    "lexical_graph": False,
                    "extraction": False,
                    "embeddings": False,
                    "similarity_edges": False,
                    "community_detection": True  # Only this enabled
                }
            }
        )
        
        plan = registry.get_execution_plan(config)
        stage_names = [s.name for s in plan]
        
        # Should auto-enable all dependencies
        assert "lexical_graph" in stage_names
        assert "extraction" in stage_names
        assert "embeddings" in stage_names
        assert "similarity_edges" in stage_names
        assert "community_detection" in stage_names
    
    def test_execution_order_respects_dependencies(self):
        """Test that stages are executed in dependency order."""
        config = Config(
            pipeline={
                "stages": {
                    "community_detection": True,
                    "summarization": True
                }
            }
        )
        
        plan = registry.get_execution_plan(config)
        stage_names = [s.name for s in plan]
        
        # Check order
        assert stage_names.index("lexical_graph") < stage_names.index("extraction")
        assert stage_names.index("extraction") < stage_names.index("embeddings")
        assert stage_names.index("embeddings") < stage_names.index("similarity_edges")
        assert stage_names.index("similarity_edges") < stage_names.index("community_detection")
        assert stage_names.index("community_detection") < stage_names.index("summarization")
    
    def test_semantic_resolution_requires_embeddings(self):
        """Test that semantic_resolution requires embeddings."""
        config = Config(
            pipeline={
                "stages": {
                    "semantic_resolution": True
                }
            }
        )
        
        plan = registry.get_execution_plan(config)
        stage_names = [s.name for s in plan]
        
        assert "embeddings" in stage_names
        assert "extraction" in stage_names
        assert "lexical_graph" in stage_names
    
    def test_independent_stages_not_auto_enabled(self):
        """Test that independent stages are not auto-enabled."""
        config = Config(
            pipeline={
                "stages": {
                    "extraction": True,
                    "schema_export": False,
                    "neo4j_upload": False
                }
            }
        )
        
        plan = registry.get_execution_plan(config)
        stage_names = [s.name for s in plan]
        
        # Should not include disabled independent stages
        assert "schema_export" not in stage_names
        assert "neo4j_upload" not in stage_names
    
    def test_minimal_pipeline(self):
        """Test minimal pipeline with just lexical_graph."""
        config = Config(
            pipeline={
                "stages": {
                    "lexical_graph": True,
                    "extraction": False
                }
            }
        )
        
        plan = registry.get_execution_plan(config)
        
        assert len(plan) == 1
        assert plan[0].name == "lexical_graph"


class TestStageRegistry:
    """Test stage registry functionality."""
    
    def test_all_stages_registered(self):
        """Test that all expected stages are registered."""
        expected_stages = [
            "lexical_graph",
            "extraction",
            "embeddings",
            "semantic_resolution",
            "similarity_edges",
            "community_detection",
            "summarization",
            "schema_export",
            "neo4j_upload"
        ]
        
        for stage_name in expected_stages:
            stage = registry.get_stage(stage_name)
            assert stage is not None
            assert isinstance(stage, PipelineStage)
    
    def test_stage_has_required_attributes(self):
        """Test that stages have required attributes."""
        stage = registry.get_stage("community_detection")
        
        assert hasattr(stage, 'name')
        assert hasattr(stage, 'display_name')
        assert hasattr(stage, 'description')
        assert hasattr(stage, 'run_func')
        assert hasattr(stage, 'depends_on')
    
    def test_dependency_chain(self):
        """Test the full dependency chain."""
        # Community detection chain
        cd_stage = registry.get_stage("community_detection")
        assert "similarity_edges" in cd_stage.depends_on
        
        # Similarity edges chain
        se_stage = registry.get_stage("similarity_edges")
        assert "embeddings" in se_stage.depends_on
        
        # Embeddings chain
        emb_stage = registry.get_stage("embeddings")
        assert "extraction" in emb_stage.depends_on
        
        # Extraction chain
        ext_stage = registry.get_stage("extraction")
        assert "lexical_graph" in ext_stage.depends_on


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
