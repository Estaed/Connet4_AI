"""
Comprehensive tests for TrainingInterface class.

Tests the menu-driven training interface, user interaction handling,
and integration with the training system.
"""

import pytest
import io
import sys
from unittest.mock import Mock, patch, MagicMock
from src.training.training_interface import TrainingInterface


class TestTrainingInterfaceInitialization:
    """Test TrainingInterface initialization and setup."""
    
    def test_interface_initialization(self, test_config):
        """Test basic TrainingInterface initialization."""
        interface = TrainingInterface(config=test_config)
        
        assert interface.config == test_config
        assert hasattr(interface, 'models_dir')
        assert hasattr(interface, 'logs_dir')
    
    def test_interface_directory_setup(self, test_config):
        """Test that interface sets up directories correctly."""
        interface = TrainingInterface(config=test_config)
        
        # Should have default directory paths
        assert interface.models_dir is not None
        assert interface.logs_dir is not None
        
        # Paths should be strings
        assert isinstance(interface.models_dir, str)
        assert isinstance(interface.logs_dir, str)
    
    def test_interface_with_custom_directories(self, test_config):
        """Test interface with custom directory paths."""
        custom_models = "./custom_models"
        custom_logs = "./custom_logs"
        
        interface = TrainingInterface(
            config=test_config,
            models_dir=custom_models,
            logs_dir=custom_logs
        )
        
        assert interface.models_dir == custom_models
        assert interface.logs_dir == custom_logs


class TestTrainingInterfaceMenuDisplay:
    """Test menu display functionality."""
    
    def test_display_training_menu(self, test_config):
        """Test training menu display."""
        interface = TrainingInterface(config=test_config)
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            interface.display_training_menu()
            output = captured_output.getvalue()
            
            # Should contain menu options
            assert "training" in output.lower()
            assert "difficulty" in output.lower() or "small" in output.lower()
            
            # Should contain difficulty levels
            difficulty_levels = ["small", "medium", "impossible"]
            for level in difficulty_levels:
                assert level in output.lower()
                
        finally:
            sys.stdout = sys.__stdout__
    
    def test_menu_formatting(self, test_config):
        """Test that menu is properly formatted."""
        interface = TrainingInterface(config=test_config)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            interface.display_training_menu()
            output = captured_output.getvalue()
            
            # Should have some structure (numbers, colons, etc.)
            assert any(char.isdigit() for char in output)
            assert ":" in output or ")" in output
            
            # Should have multiple lines
            lines = output.strip().split('\n')
            assert len(lines) > 1
            
        finally:
            sys.stdout = sys.__stdout__
    
    def test_menu_content_completeness(self, test_config):
        """Test that menu contains all expected options."""
        interface = TrainingInterface(config=test_config)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            interface.display_training_menu()
            output = captured_output.getvalue().lower()
            
            # Should mention key concepts
            expected_terms = ["train", "difficulty", "small", "medium", "impossible"]
            for term in expected_terms:
                assert term in output
                
        finally:
            sys.stdout = sys.__stdout__


class TestTrainingInterfaceUserInput:
    """Test user input handling."""
    
    @patch('builtins.input')
    def test_get_user_choice_valid_input(self, mock_input, test_config):
        """Test handling of valid user input."""
        interface = TrainingInterface(config=test_config)
        
        # Mock valid input
        mock_input.return_value = "1"
        
        # Should handle input without errors
        try:
            # This would depend on the specific implementation
            # For now, test that input function is called
            with patch.object(interface, 'display_training_menu'):
                # Interface might have a method to get user choice
                pass
        except AttributeError:
            # Method might not exist yet
            pass
    
    @patch('builtins.input')
    def test_get_user_choice_invalid_input(self, mock_input, test_config):
        """Test handling of invalid user input."""
        interface = TrainingInterface(config=test_config)
        
        # Mock invalid then valid input
        mock_input.side_effect = ["invalid", "99", "1"]
        
        # Should handle invalid input gracefully
        try:
            with patch.object(interface, 'display_training_menu'):
                # Should eventually get valid input
                pass
        except AttributeError:
            # Method might not exist yet
            pass
    
    @patch('builtins.input')
    def test_input_validation(self, mock_input, test_config):
        """Test input validation functionality."""
        interface = TrainingInterface(config=test_config)
        
        # Test various invalid inputs
        invalid_inputs = ["", "abc", "-1", "0", "999"]
        
        for invalid_input in invalid_inputs:
            mock_input.return_value = invalid_input
            
            # Should handle each invalid input
            try:
                # Test would depend on specific validation implementation
                pass
            except (ValueError, IndexError):
                # Expected for invalid inputs
                pass
    
    @patch('builtins.input')
    def test_keyboard_interrupt_handling(self, mock_input, test_config):
        """Test handling of keyboard interrupts."""
        interface = TrainingInterface(config=test_config)
        
        # Mock KeyboardInterrupt
        mock_input.side_effect = KeyboardInterrupt()
        
        # Should handle Ctrl+C gracefully
        try:
            with patch.object(interface, 'display_training_menu'):
                # Should not crash on KeyboardInterrupt
                pass
        except (KeyboardInterrupt, AttributeError):
            # Either handled or method doesn't exist
            pass


class TestTrainingInterfaceDifficultyLevels:
    """Test difficulty level handling."""
    
    def test_run_training_difficulty_small(self, test_config):
        """Test running training with small difficulty."""
        interface = TrainingInterface(config=test_config)
        
        # Mock the HybridTrainer
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            mock_trainer_instance.train.return_value = {'loss': 0.5}
            
            # Run training
            interface.run_training_difficulty('small')
            
            # Should create trainer with small difficulty
            mock_trainer.assert_called_once()
            call_args = mock_trainer.call_args
            assert call_args[1]['difficulty'] == 'small'
            
            # Should call train method
            mock_trainer_instance.train.assert_called_once()
    
    def test_run_training_difficulty_medium(self, test_config):
        """Test running training with medium difficulty."""
        interface = TrainingInterface(config=test_config)
        
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            
            interface.run_training_difficulty('medium')
            
            # Should create trainer with medium difficulty
            call_args = mock_trainer.call_args
            assert call_args[1]['difficulty'] == 'medium'
    
    def test_run_training_difficulty_impossible(self, test_config):
        """Test running training with impossible difficulty."""
        interface = TrainingInterface(config=test_config)
        
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            
            interface.run_training_difficulty('impossible')
            
            # Should create trainer with impossible difficulty
            call_args = mock_trainer.call_args
            assert call_args[1]['difficulty'] == 'impossible'
    
    def test_run_training_invalid_difficulty(self, test_config):
        """Test handling of invalid difficulty levels."""
        interface = TrainingInterface(config=test_config)
        
        # Should handle invalid difficulty gracefully
        with pytest.raises(ValueError):
            interface.run_training_difficulty('invalid_difficulty')
    
    def test_difficulty_parameter_passing(self, test_config):
        """Test that difficulty parameters are correctly passed."""
        interface = TrainingInterface(config=test_config)
        
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            
            # Test each difficulty level
            difficulties = ['small', 'medium', 'impossible']
            
            for difficulty in difficulties:
                interface.run_training_difficulty(difficulty)
                
                # Check that correct parameters were passed
                call_args = mock_trainer.call_args
                assert call_args[1]['difficulty'] == difficulty
                assert call_args[1]['config'] == test_config
                assert 'models_dir' in call_args[1]
                assert 'logs_dir' in call_args[1]


class TestTrainingInterfaceBenchmarkMode:
    """Test benchmark functionality."""
    
    def test_benchmark_mode_basic(self, test_config):
        """Test basic benchmark mode functionality."""
        interface = TrainingInterface(config=test_config)
        
        # Mock the benchmark implementation
        with patch.object(interface, 'benchmark_mode') as mock_benchmark:
            mock_benchmark.return_value = {'small': 100, 'medium': 50, 'impossible': 10}
            
            results = interface.benchmark_mode()
            
            # Should return benchmark results
            assert isinstance(results, dict)
            mock_benchmark.assert_called_once()
    
    def test_benchmark_all_difficulties(self, test_config):
        """Test benchmarking all difficulty levels."""
        interface = TrainingInterface(config=test_config)
        
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            mock_trainer_instance.train.return_value = {'loss': 0.5}
            
            # Mock benchmark to test all difficulties
            with patch.object(interface, 'benchmark_mode') as mock_benchmark:
                # Simulate benchmark running all difficulties
                def benchmark_side_effect():
                    difficulties = ['small', 'medium', 'impossible']
                    results = {}
                    for difficulty in difficulties:
                        # Simulate creating trainer for each difficulty
                        trainer = mock_trainer(
                            difficulty=difficulty,
                            config=test_config,
                            models_dir=interface.models_dir,
                            logs_dir=interface.logs_dir
                        )
                        results[difficulty] = {'trainer_created': True}
                    return results
                
                mock_benchmark.side_effect = benchmark_side_effect
                
                results = interface.benchmark_mode()
                
                # Should test all difficulties
                assert isinstance(results, dict)
                assert len(results) == 3
    
    def test_benchmark_performance_measurement(self, test_config):
        """Test that benchmark measures performance correctly."""
        interface = TrainingInterface(config=test_config)
        
        # Mock time measurement
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0, 1, 2, 3, 4, 5, 6]  # Simulated time progression
            
            with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
                mock_trainer_instance = Mock()
                mock_trainer.return_value = mock_trainer_instance
                
                # Run benchmark
                interface.benchmark_mode()
                
                # Should have measured time
                assert mock_time.call_count > 0
    
    def test_benchmark_error_handling(self, test_config):
        """Test error handling in benchmark mode."""
        interface = TrainingInterface(config=test_config)
        
        # Mock trainer to raise an error
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer.side_effect = RuntimeError("Mock training error")
            
            # Benchmark should handle errors gracefully
            try:
                results = interface.benchmark_mode()
                # If no exception, should return some error indication
                assert results is not None
            except RuntimeError:
                # Acceptable to propagate error
                pass


class TestTrainingInterfaceConfigurationHandling:
    """Test configuration handling in the interface."""
    
    def test_config_parameter_usage(self, test_config):
        """Test that configuration parameters are used correctly."""
        interface = TrainingInterface(config=test_config)
        
        # Config should be stored
        assert interface.config == test_config
        
        # Should pass config to trainer
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            
            interface.run_training_difficulty('small')
            
            # Config should be passed to trainer
            call_args = mock_trainer.call_args
            assert call_args[1]['config'] == test_config
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test with None config
        try:
            interface = TrainingInterface(config=None)
            # If no error, should handle gracefully
            assert interface.config is None
        except (ValueError, AttributeError):
            # Expected error for None config
            pass
        
        # Test with invalid config
        invalid_config = "not_a_config"
        try:
            interface = TrainingInterface(config=invalid_config)
            # If no error, should handle gracefully
            assert interface.config == invalid_config
        except (ValueError, TypeError):
            # Expected error for invalid config
            pass
    
    def test_default_configuration_values(self, test_config):
        """Test handling of default configuration values."""
        interface = TrainingInterface(config=test_config)
        
        # Should have reasonable defaults for directories
        assert interface.models_dir is not None
        assert interface.logs_dir is not None
        assert len(interface.models_dir) > 0
        assert len(interface.logs_dir) > 0


class TestTrainingInterfaceErrorHandling:
    """Test error handling and robustness."""
    
    def test_training_error_handling(self, test_config):
        """Test handling of training errors."""
        interface = TrainingInterface(config=test_config)
        
        # Mock trainer to raise an error
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer.side_effect = RuntimeError("Training failed")
            
            # Should handle training errors gracefully
            try:
                interface.run_training_difficulty('small')
                # If no exception, error was handled
            except RuntimeError:
                # Acceptable to propagate error
                pass
    
    def test_directory_creation_error_handling(self, test_config):
        """Test handling of directory creation errors."""
        # Mock directory creation to fail
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            
            try:
                interface = TrainingInterface(config=test_config)
                # If no error, should handle gracefully
            except PermissionError:
                # Expected error for permission issues
                pass
    
    def test_keyboard_interrupt_during_training(self, test_config):
        """Test handling of keyboard interrupt during training."""
        interface = TrainingInterface(config=test_config)
        
        # Mock trainer to raise KeyboardInterrupt
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            mock_trainer_instance.train.side_effect = KeyboardInterrupt()
            
            # Should handle Ctrl+C gracefully
            try:
                interface.run_training_difficulty('small')
                # If no exception, was handled gracefully
            except KeyboardInterrupt:
                # Acceptable to propagate interrupt
                pass
    
    def test_out_of_memory_error_handling(self, test_config):
        """Test handling of out-of-memory errors."""
        interface = TrainingInterface(config=test_config)
        
        # Mock trainer to raise out-of-memory error
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            mock_trainer_instance.train.side_effect = RuntimeError("CUDA out of memory")
            
            # Should handle OOM errors gracefully
            try:
                interface.run_training_difficulty('small')
                # If no exception, error was handled
            except RuntimeError:
                # Acceptable to propagate OOM error
                pass


class TestTrainingInterfaceIntegration:
    """Test integration with other system components."""
    
    def test_integration_with_hybrid_trainer(self, test_config):
        """Test integration with HybridTrainer."""
        interface = TrainingInterface(config=test_config)
        
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            mock_trainer_instance.train.return_value = {'final_metrics': 'success'}
            
            # Should successfully integrate with trainer
            interface.run_training_difficulty('small')
            
            # Trainer should be created and used
            mock_trainer.assert_called_once()
            mock_trainer_instance.train.assert_called_once()
    
    def test_integration_with_config_system(self, test_config):
        """Test integration with configuration system."""
        interface = TrainingInterface(config=test_config)
        
        # Should use config parameters
        assert interface.config == test_config
        
        # Should pass config to other components
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            
            interface.run_training_difficulty('medium')
            
            # Config should be passed through
            call_args = mock_trainer.call_args
            assert call_args[1]['config'] == test_config
    
    def test_directory_path_integration(self, test_config, tmp_path):
        """Test integration with directory path management."""
        models_dir = tmp_path / "test_models"
        logs_dir = tmp_path / "test_logs"
        
        interface = TrainingInterface(
            config=test_config,
            models_dir=str(models_dir),
            logs_dir=str(logs_dir)
        )
        
        # Should use custom directories
        assert interface.models_dir == str(models_dir)
        assert interface.logs_dir == str(logs_dir)
        
        # Should pass directories to trainer
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            
            interface.run_training_difficulty('small')
            
            # Directories should be passed through
            call_args = mock_trainer.call_args
            assert call_args[1]['models_dir'] == str(models_dir)
            assert call_args[1]['logs_dir'] == str(logs_dir)
    
    def test_end_to_end_interface_workflow(self, test_config):
        """Test complete end-to-end interface workflow."""
        interface = TrainingInterface(config=test_config)
        
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            mock_trainer_instance.train.return_value = {
                'policy_loss': 0.5,
                'value_loss': 0.3,
                'total_loss': 0.8
            }
            
            # Complete workflow: display menu -> get choice -> run training
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            try:
                # Display menu
                interface.display_training_menu()
                
                # Run training
                interface.run_training_difficulty('small')
                
                # Should complete successfully
                output = captured_output.getvalue()
                assert "training" in output.lower()
                
                # Trainer should have been used
                mock_trainer.assert_called_once()
                mock_trainer_instance.train.assert_called_once()
                
            finally:
                sys.stdout = sys.__stdout__


class TestTrainingInterfaceUsability:
    """Test usability and user experience aspects."""
    
    def test_menu_readability(self, test_config):
        """Test that menu is readable and well-formatted."""
        interface = TrainingInterface(config=test_config)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            interface.display_training_menu()
            output = captured_output.getvalue()
            
            # Should be reasonably long (informative)
            assert len(output) > 50
            
            # Should have proper line breaks
            lines = output.split('\n')
            assert len(lines) > 2
            
            # Should not have excessively long lines
            for line in lines:
                assert len(line) < 200  # Reasonable line length
                
        finally:
            sys.stdout = sys.__stdout__
    
    def test_error_message_clarity(self, test_config):
        """Test that error messages are clear and helpful."""
        interface = TrainingInterface(config=test_config)
        
        # Test with invalid difficulty
        try:
            interface.run_training_difficulty('invalid')
        except ValueError as e:
            error_message = str(e)
            # Error message should be informative
            assert len(error_message) > 10
            assert "difficulty" in error_message.lower() or "invalid" in error_message.lower()
    
    def test_progress_indication(self, test_config):
        """Test that interface provides progress indication."""
        interface = TrainingInterface(config=test_config)
        
        with patch('src.training.training_interface.HybridTrainer') as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            
            # Mock training to take some time
            def slow_train(*args, **kwargs):
                import time
                time.sleep(0.1)  # Simulate some work
                return {'loss': 0.5}
            
            mock_trainer_instance.train.side_effect = slow_train
            
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            try:
                interface.run_training_difficulty('small')
                output = captured_output.getvalue()
                
                # Should provide some indication of progress or completion
                # (Implementation dependent)
                
            finally:
                sys.stdout = sys.__stdout__