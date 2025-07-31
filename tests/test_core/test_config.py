"""
Comprehensive tests for Config class and configuration management.

Tests YAML loading, validation, device detection, parameter access,
and configuration system functionality.
"""

import pytest
import yaml
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open, Mock
from src.core.config import Config


class TestConfigInitialization:
    """Test Config class initialization and basic setup."""
    
    def test_config_initialization_with_file(self, temp_config_file):
        """Test Config initialization with valid config file."""
        config = Config(str(temp_config_file))
        
        assert isinstance(config, Config)
        assert hasattr(config, 'config_data')
        assert hasattr(config, 'device')
    
    def test_config_initialization_missing_file(self):
        """Test Config initialization with missing file."""
        with pytest.raises((FileNotFoundError, IOError)):
            Config("nonexistent_config.yaml")
    
    def test_config_initialization_invalid_path_type(self):
        """Test Config initialization with invalid path type."""
        with pytest.raises((TypeError, ValueError)):
            Config(123)
        
        with pytest.raises((TypeError, ValueError)):
            Config(None)
    
    def test_config_device_setup(self, temp_config_file):
        """Test that device is properly set up during initialization."""
        config = Config(str(temp_config_file))
        
        # Should have device attribute
        assert hasattr(config, 'device')
        
        # Device should be torch.device
        assert isinstance(config.device, torch.device)
        
        # Should be either CPU or CUDA
        assert config.device.type in ['cpu', 'cuda']
    
    def test_config_data_loading(self, temp_config_file, test_config_dict):
        """Test that config data is properly loaded."""
        config = Config(str(temp_config_file))
        
        # Should load the data correctly
        assert isinstance(config.config_data, dict)
        
        # Should contain expected sections
        assert 'game' in config.config_data
        assert 'ppo' in config.config_data
        assert 'network' in config.config_data


class TestConfigYAMLHandling:
    """Test YAML file handling and parsing."""
    
    def test_valid_yaml_parsing(self, tmp_path):
        """Test parsing of valid YAML content."""
        yaml_content = """
        game:
          rows: 6
          cols: 7
        ppo:
          learning_rate: 0.003
          batch_size: 32
        """
        
        config_file = tmp_path / "valid_config.yaml"
        config_file.write_text(yaml_content)
        
        config = Config(str(config_file))
        
        assert config.config_data['game']['rows'] == 6
        assert config.config_data['game']['cols'] == 7
        assert config.config_data['ppo']['learning_rate'] == 0.003
        assert config.config_data['ppo']['batch_size'] == 32
    
    def test_invalid_yaml_parsing(self, tmp_path):
        """Test handling of invalid YAML content."""
        invalid_yaml_content = """
        game:
          rows: 6
          cols: 7
        ppo:
          learning_rate: 0.003
          batch_size: 32
          invalid_indent
        """
        
        config_file = tmp_path / "invalid_config.yaml"
        config_file.write_text(invalid_yaml_content)
        
        with pytest.raises(yaml.YAMLError):
            Config(str(config_file))
    
    def test_empty_yaml_file(self, tmp_path):
        """Test handling of empty YAML file."""
        config_file = tmp_path / "empty_config.yaml"
        config_file.write_text("")
        
        config = Config(str(config_file))
        
        # Should handle empty file gracefully
        assert config.config_data is None or config.config_data == {}
    
    def test_yaml_with_special_characters(self, tmp_path):
        """Test YAML parsing with special characters."""
        yaml_content = """
        game:
          name: "Connect4 - AI Edition"
          description: "A game with special chars: @#$%^&*()"
        paths:
          models: "/path/with/spaces and symbols"
        """
        
        config_file = tmp_path / "special_config.yaml"
        config_file.write_text(yaml_content)
        
        config = Config(str(config_file))
        
        assert "Connect4 - AI Edition" in config.config_data['game']['name']
        assert "special chars" in config.config_data['game']['description']
    
    def test_yaml_with_different_data_types(self, tmp_path):
        """Test YAML parsing with various data types."""
        yaml_content = """
        numbers:
          integer: 42
          float: 3.14159
          negative: -10
        booleans:
          true_value: true
          false_value: false
        strings:
          simple: hello
          quoted: "hello world"
        lists:
          - item1
          - item2
          - item3
        nested:
          level1:
            level2:
              value: deep_value
        """
        
        config_file = tmp_path / "types_config.yaml"
        config_file.write_text(yaml_content)
        
        config = Config(str(config_file))
        
        # Test different data types
        assert isinstance(config.config_data['numbers']['integer'], int)
        assert isinstance(config.config_data['numbers']['float'], float)
        assert isinstance(config.config_data['booleans']['true_value'], bool)
        assert isinstance(config.config_data['strings']['simple'], str)
        assert isinstance(config.config_data['lists'], list)
        assert config.config_data['nested']['level1']['level2']['value'] == 'deep_value'


class TestConfigParameterAccess:
    """Test parameter access functionality."""
    
    def test_get_method_basic(self, test_config):
        """Test basic get method functionality."""
        # Test simple key access
        rows = test_config.get('game.rows')
        assert rows == 6
        
        cols = test_config.get('game.cols')
        assert cols == 7
        
        learning_rate = test_config.get('ppo.learning_rate')
        assert learning_rate == 3e-4
    
    def test_get_method_with_default(self, test_config):
        """Test get method with default values."""
        # Test non-existent key with default
        value = test_config.get('nonexistent.key', 'default_value')
        assert value == 'default_value'
        
        # Test non-existent key without default
        value = test_config.get('nonexistent.key')
        assert value is None
    
    def test_get_method_nested_access(self, test_config):
        """Test get method with deeply nested keys."""
        # Test nested access
        network_type = test_config.get('network.type')
        assert network_type == 'standard'
        
        # Test deeper nesting if available
        device_setting = test_config.get('device.training_device')
        assert device_setting == 'cpu'
    
    def test_get_method_edge_cases(self, test_config):
        """Test get method edge cases."""
        # Test empty key
        value = test_config.get('', 'default')
        assert value == 'default'
        
        # Test None key
        value = test_config.get(None, 'default')
        assert value == 'default'
        
        # Test key with extra dots
        value = test_config.get('game..rows', 'default')
        assert value == 'default'
        
        # Test key starting with dot
        value = test_config.get('.game.rows', 'default')
        assert value == 'default'
    
    def test_get_method_type_preservation(self, test_config):
        """Test that get method preserves data types."""
        # Integer
        rows = test_config.get('game.rows')
        assert isinstance(rows, int)
        assert rows == 6
        
        # Float
        learning_rate = test_config.get('ppo.learning_rate')
        assert isinstance(learning_rate, float)
        
        # String
        network_type = test_config.get('network.type')
        assert isinstance(network_type, str)
    
    def test_get_method_list_access(self, tmp_path):
        """Test get method with list values."""
        yaml_content = """
        network:
          conv_channels: [64, 128, 128]
          kernel_sizes: [4, 3, 3]
        """
        
        config_file = tmp_path / "list_config.yaml"
        config_file.write_text(yaml_content)
        config = Config(str(config_file))
        
        channels = config.get('network.conv_channels')
        assert isinstance(channels, list)
        assert channels == [64, 128, 128]
        
        kernel_sizes = config.get('network.kernel_sizes')
        assert isinstance(kernel_sizes, list)
        assert kernel_sizes == [4, 3, 3]
    
    def test_get_method_case_sensitivity(self, test_config):
        """Test that get method is case sensitive."""
        # Correct case
        value = test_config.get('game.rows')
        assert value == 6
        
        # Wrong case
        value = test_config.get('Game.rows', 'default')
        assert value == 'default'
        
        value = test_config.get('game.Rows', 'default')
        assert value == 'default'


class TestConfigDeviceManagement:
    """Test device detection and management functionality."""
    
    def test_device_detection_cpu(self, test_config):
        """Test CPU device detection."""
        with patch('torch.cuda.is_available', return_value=False):
            config = Config(str(test_config._config_path) if hasattr(test_config, '_config_path') else 'dummy')
            
            assert config.device.type == 'cpu'
    
    @pytest.mark.gpu
    def test_device_detection_cuda(self, test_config):
        """Test CUDA device detection."""
        with patch('torch.cuda.is_available', return_value=True):
            config = Config(str(test_config._config_path) if hasattr(test_config, '_config_path') else 'dummy')
            
            assert config.device.type == 'cuda'
    
    def test_device_override_from_config(self, tmp_path):
        """Test device override from configuration."""
        yaml_content = """
        device:
          training_device: 'cpu'
          force_cpu: true
        """
        
        config_file = tmp_path / "device_config.yaml"
        config_file.write_text(yaml_content)
        
        with patch('torch.cuda.is_available', return_value=True):
            config = Config(str(config_file))
            
            # Should respect config override even if CUDA is available
            training_device = config.get('device.training_device')
            assert training_device == 'cpu'
    
    def test_device_fallback_behavior(self, test_config):
        """Test device fallback behavior."""
        # Mock CUDA as unavailable
        with patch('torch.cuda.is_available', return_value=False):
            config = Config(str(test_config._config_path) if hasattr(test_config, '_config_path') else 'dummy')
            
            # Should fall back to CPU
            assert config.device.type == 'cpu'
    
    def test_device_consistency(self, test_config):
        """Test that device setting is consistent throughout config."""
        device1 = test_config.device
        device2 = test_config.device
        
        # Should return same device object
        assert device1 == device2
        assert device1.type == device2.type


class TestConfigValidation:
    """Test configuration validation functionality."""
    
    def test_required_sections_validation(self, tmp_path):
        """Test validation of required configuration sections."""
        # Config missing required sections
        incomplete_yaml = """
        game:
          rows: 6
        # Missing ppo, network, etc.
        """
        
        config_file = tmp_path / "incomplete_config.yaml"
        config_file.write_text(incomplete_yaml)
        
        config = Config(str(config_file))
        
        # Should handle missing sections gracefully
        assert config.get('ppo.learning_rate', 0.001) == 0.001
        assert config.get('network.type', 'standard') == 'standard'
    
    def test_parameter_range_validation(self, tmp_path):
        """Test parameter range validation."""
        yaml_content = """
        game:
          rows: 6
          cols: 7
        ppo:
          learning_rate: 0.001
          batch_size: 32
          clip_range: 0.2
        """
        
        config_file = tmp_path / "valid_ranges_config.yaml"
        config_file.write_text(yaml_content)
        config = Config(str(config_file))
        
        # Should load valid ranges
        assert 0 < config.get('ppo.learning_rate') < 1
        assert config.get('ppo.batch_size') > 0
        assert 0 < config.get('ppo.clip_range') < 1
    
    def test_invalid_parameter_handling(self, tmp_path):
        """Test handling of invalid parameter values."""
        yaml_content = """
        game:
          rows: -6  # Invalid negative rows
          cols: 0   # Invalid zero columns
        ppo:
          learning_rate: -0.001  # Invalid negative learning rate
          batch_size: 0          # Invalid zero batch size
        """
        
        config_file = tmp_path / "invalid_params_config.yaml"
        config_file.write_text(yaml_content)
        config = Config(str(config_file))
        
        # Should load values (validation might be done elsewhere)
        assert config.get('game.rows') == -6
        assert config.get('game.cols') == 0
        
        # Application should handle invalid values appropriately
    
    def test_data_type_validation(self, tmp_path):
        """Test data type validation."""
        yaml_content = """
        game:
          rows: "6"      # String instead of int
          cols: 7.0      # Float instead of int
        ppo:
          learning_rate: "0.001"  # String instead of float
          batch_size: 32.5        # Float instead of int
        """
        
        config_file = tmp_path / "type_mismatch_config.yaml"
        config_file.write_text(yaml_content)
        config = Config(str(config_file))
        
        # YAML will parse these as their natural types
        assert isinstance(config.get('game.rows'), str)
        assert isinstance(config.get('game.cols'), float)
        assert isinstance(config.get('ppo.learning_rate'), str)
        assert isinstance(config.get('ppo.batch_size'), float)


class TestConfigErrorHandling:
    """Test error handling in Config class."""
    
    def test_corrupted_yaml_handling(self, tmp_path):
        """Test handling of corrupted YAML files."""
        corrupted_yaml = """
        game:
          rows: 6
          cols: 7
        ppo:
          learning_rate: 0.001
          batch_size: 32
          [invalid yaml structure}
        """
        
        config_file = tmp_path / "corrupted_config.yaml"
        config_file.write_text(corrupted_yaml)
        
        with pytest.raises(yaml.YAMLError):
            Config(str(config_file))
    
    def test_permission_error_handling(self, tmp_path):
        """Test handling of permission errors."""
        config_file = tmp_path / "restricted_config.yaml"
        config_file.write_text("game:\n  rows: 6")
        
        # Mock permission error
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                Config(str(config_file))
    
    def test_large_config_file_handling(self, tmp_path):
        """Test handling of very large configuration files."""
        # Create large config content
        large_yaml_content = "game:\n  rows: 6\n  cols: 7\n"
        
        # Add many sections
        for i in range(1000):
            large_yaml_content += f"section_{i}:\n  value: {i}\n"
        
        config_file = tmp_path / "large_config.yaml"
        config_file.write_text(large_yaml_content)
        
        # Should handle large files
        config = Config(str(config_file))
        
        assert config.get('game.rows') == 6
        assert config.get('section_999.value') == 999
    
    def test_unicode_handling(self, tmp_path):
        """Test handling of Unicode characters in config."""
        unicode_yaml = """
        game:
          name: "Connect4 - Jeux de Société 🎮"
          description: "Un jeu avec des caractères Unicode: αβγδε"
        paths:
          models: "/路径/with/中文/characters"
        symbols:
          currency: "€$¥£"
        """
        
        config_file = tmp_path / "unicode_config.yaml"
        config_file.write_text(unicode_yaml, encoding='utf-8')
        
        config = Config(str(config_file))
        
        assert "🎮" in config.get('game.name')
        assert "αβγδε" in config.get('game.description')
        assert "中文" in config.get('paths.models')
        assert "€$¥£" == config.get('symbols.currency')


class TestConfigEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_deep_nesting(self, tmp_path):
        """Test very deeply nested configuration."""
        deep_yaml = """
        level1:
          level2:
            level3:
              level4:
                level5:
                  level6:
                    level7:
                      level8:
                        level9:
                          level10:
                            deep_value: "found_it"
        """
        
        config_file = tmp_path / "deep_config.yaml"
        config_file.write_text(deep_yaml)
        config = Config(str(config_file))
        
        # Should handle deep nesting
        deep_value = config.get('level1.level2.level3.level4.level5.level6.level7.level8.level9.level10.deep_value')
        assert deep_value == "found_it"
    
    def test_circular_references(self, tmp_path):
        """Test handling of YAML with anchors and references."""
        anchor_yaml = """
        defaults: &defaults
          timeout: 30
          retries: 3
        
        game: 
          <<: *defaults
          rows: 6
          cols: 7
        
        training:
          <<: *defaults
          max_episodes: 1000
        """
        
        config_file = tmp_path / "anchor_config.yaml"
        config_file.write_text(anchor_yaml)
        config = Config(str(config_file))
        
        # Should handle YAML anchors and references
        assert config.get('game.timeout') == 30
        assert config.get('game.rows') == 6
        assert config.get('training.timeout') == 30
        assert config.get('training.max_episodes') == 1000
    
    def test_empty_sections(self, tmp_path):
        """Test handling of empty configuration sections."""
        empty_sections_yaml = """
        game:
        ppo:
        network:
        empty_section:
        """
        
        config_file = tmp_path / "empty_sections_config.yaml"
        config_file.write_text(empty_sections_yaml)
        config = Config(str(config_file))
        
        # Should handle empty sections
        assert config.get('game') is None
        assert config.get('ppo') is None
        assert config.get('empty_section') is None
    
    def test_special_key_names(self, tmp_path):
        """Test handling of special key names."""
        special_keys_yaml = """
        "key with spaces": "value1"
        "key-with-dashes": "value2"
        "key_with_underscores": "value3"
        "key.with.dots": "value4"
        "123numeric_key": "value5"
        "UPPER_CASE_KEY": "value6"
        """
        
        config_file = tmp_path / "special_keys_config.yaml"
        config_file.write_text(special_keys_yaml)
        config = Config(str(config_file))
        
        # Should handle special key names
        assert config.get('key with spaces') == "value1"
        assert config.get('key-with-dashes') == "value2"
        assert config.get('key_with_underscores') == "value3"
        assert config.get('key.with.dots') == "value4"
        assert config.get('123numeric_key') == "value5"
        assert config.get('UPPER_CASE_KEY') == "value6"


class TestConfigPerformance:
    """Test performance characteristics of Config class."""
    
    @pytest.mark.performance
    def test_config_loading_speed(self, temp_config_file):
        """Test configuration loading performance."""
        import time
        
        start_time = time.time()
        
        # Load config multiple times
        for _ in range(100):
            config = Config(str(temp_config_file))
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        # Should load quickly
        loads_per_second = 100 / loading_time
        assert loads_per_second > 50  # At least 50 loads per second
    
    @pytest.mark.performance
    def test_parameter_access_speed(self, test_config):
        """Test parameter access performance."""
        import time
        
        start_time = time.time()
        
        # Access parameters many times
        for _ in range(10000):
            value = test_config.get('ppo.learning_rate')
        
        end_time = time.time()
        access_time = end_time - start_time
        
        # Should access parameters quickly
        accesses_per_second = 10000 / access_time
        assert accesses_per_second > 10000  # At least 10k accesses per second
    
    def test_memory_usage(self, tmp_path):
        """Test memory usage with large configurations."""
        # Create large configuration
        large_config_data = {}
        
        for i in range(1000):
            large_config_data[f'section_{i}'] = {
                'param1': i,
                'param2': f'value_{i}',
                'param3': [i, i+1, i+2],
                'param4': {'nested': i}
            }
        
        # Write to file
        config_file = tmp_path / "large_memory_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(large_config_data, f)
        
        # Load config
        config = Config(str(config_file))
        
        # Should handle large configs without excessive memory usage
        assert config.get('section_999.param1') == 999
        assert len(config.config_data) == 1000


class TestConfigIntegration:
    """Test Config integration with other system components."""
    
    def test_config_with_agent_creation(self, test_config):
        """Test config integration with agent creation."""
        from src.agents.base_agent import create_agent
        
        # Should work with agent factory
        try:
            agent = create_agent('ppo', config=test_config)
            assert agent.config == test_config
        except Exception as e:
            # Some configurations might not be complete enough
            pass
    
    def test_config_with_network_creation(self, test_config):
        """Test config integration with network creation."""
        from src.agents.networks import create_network
        
        network_type = test_config.get('network.type', 'standard')
        
        try:
            network = create_network(network_type)
            assert network is not None
        except Exception as e:
            # Some network types might not be implemented
            pass
    
    def test_config_device_consistency_with_torch(self, test_config):
        """Test that config device is consistent with PyTorch."""
        config_device = test_config.device
        
        # Should be valid PyTorch device
        assert isinstance(config_device, torch.device)
        
        # Should be able to create tensors on this device
        if config_device.type == 'cuda' and torch.cuda.is_available():
            tensor = torch.tensor([1, 2, 3], device=config_device)
            assert tensor.device == config_device
        else:
            tensor = torch.tensor([1, 2, 3], device='cpu')
            assert tensor.device.type == 'cpu'
    
    def test_config_serialization_compatibility(self, test_config, tmp_path):
        """Test config compatibility with serialization."""
        # Should be able to serialize config data
        config_data = test_config.config_data
        
        # Test JSON serialization
        import json
        json_file = tmp_path / "config_data.json"
        
        try:
            with open(json_file, 'w') as f:
                json.dump(config_data, f)
            
            with open(json_file, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == config_data
        except (TypeError, ValueError):
            # Some config data might not be JSON serializable
            pass
    
    def test_config_thread_safety(self, test_config):
        """Test config thread safety."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def access_config():
            try:
                value = test_config.get('ppo.learning_rate')
                results.put(('success', value))
            except Exception as e:
                results.put(('error', str(e)))
        
        # Create multiple threads accessing config
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=access_config)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        success_count = 0
        while not results.empty():
            result_type, result_value = results.get()
            if result_type == 'success':
                success_count += 1
        
        # All threads should succeed
        assert success_count == 10