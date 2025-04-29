import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from lbw_predictor import LBWPredictor, TrajectoryPoint, InputData, ContactPosition, OutputData, Verdict

class TestLBWPredictor(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.lbw_predictor = LBWPredictor()    
    
    def test_predict_path_calculates_trajectory(self):
        """Test predict_path correctly calculates ball trajectory after impact"""
        # Create mock input data
        input_data = InputData(
            trajectory=[
                TrajectoryPoint(pos_x=0.0, pos_y=1.0, pos_z=20.0, timestamp=0.0)
            ],
            velocity_vector=[5.0, -1.0, -10.0],  # Moving towards stumps
            leg_contact_position=ContactPosition(x=0.0, y=0.5, z=15.0),
            edge_detected=False,
            decision_flag=[None]
        )
        
        # Mock the physics engine methods
        with patch.object(self.lbw_predictor.physics, 'estimate_bounce_velocity') as mock_bounce:
            mock_bounce.return_value = np.array([5.0, -1.0, -8.0])  # Slightly reduced velocity after impact
            
            with patch.object(self.lbw_predictor.physics, 'predict_trajectory') as mock_predict:
                mock_states = [
                    MagicMock(position=np.array([0.0, 0.4, 14.0]), timestamp=0.1),
                    MagicMock(position=np.array([0.5, 0.3, 13.0]), timestamp=0.2)
                ]
                mock_predict.return_value = mock_states
                
                # Call the method
                result = self.lbw_predictor.predict_path(input_data)
                
                # Verify the result
                self.assertEqual(len(result), 2)
                self.assertEqual(result[0].pos_x, 0.0)
                self.assertEqual(result[0].pos_y, 0.4)
                self.assertEqual(result[0].pos_z, 14.0)
                self.assertEqual(result[1].pos_x, 0.5)
                self.assertEqual(result[1].pos_y, 0.3)
                self.assertEqual(result[1].pos_z, 13.0)
                
    def test_check_stump_collision_detects_hit(self):
        """Test check_stump_collision correctly detects when ball hits stumps"""
        # Mock configuration
        self.lbw_predictor.config.stump_dimensions.width = 5.0
        self.lbw_predictor.config.stump_dimensions.height = 30.0
        self.lbw_predictor.config.stump_dimensions.bail_height = 2.0
        self.lbw_predictor.config.stump_dimensions.depth = 4.0
        self.lbw_predictor.config.pitch_dimensions.length = 40.0
        
        # Create a trajectory that passes through the stumps
        predicted_path = [
            TrajectoryPoint(pos_x=10.0, pos_y=10.0, pos_z=10.0, timestamp=0.0),  # Away from stumps
            TrajectoryPoint(pos_x=2.5, pos_y=15.0, pos_z=20.0, timestamp=0.1)  # Hit middle of stumps
        ]
        
        # Mock confidence calculation
        with patch.object(self.lbw_predictor, '_calculate_confidence') as mock_confidence:
            mock_confidence.return_value = 0.8
            
            # Check collision
            hit, region, confidence = self.lbw_predictor.check_stump_collision(predicted_path)
            
            # Verify results
            self.assertTrue(hit)
            self.assertEqual(region, "middle")  # Should hit the middle region
            self.assertEqual(confidence, 0.8)
    
    def test_check_stump_collision_detects_miss(self):
        """Test check_stump_collision correctly detects when ball misses stumps"""
        # Mock configuration
        self.lbw_predictor.config.stump_dimensions.width = 5.0
        self.lbw_predictor.config.stump_dimensions.height = 30.0
        self.lbw_predictor.config.stump_dimensions.bail_height = 2.0
        self.lbw_predictor.config.stump_dimensions.depth = 4.0
        self.lbw_predictor.config.pitch_dimensions.length = 40.0
        
        # Create a trajectory that misses the stumps
        predicted_path = [
            TrajectoryPoint(pos_x=10.0, pos_y=10.0, pos_z=10.0, timestamp=0.0),
            TrajectoryPoint(pos_x=10.0, pos_y=10.0, pos_z=20.0, timestamp=0.1)  # Passes to the side
        ]
        
        # Check collision
        hit, region, confidence = self.lbw_predictor.check_stump_collision(predicted_path)
        
        # Verify results
        self.assertFalse(hit)
        self.assertEqual(region, "miss")
        self.assertEqual(confidence, 0.0)
        
    def test_check_stump_collision_detects_different_impact_regions(self):
        """Test check_stump_collision correctly identifies different impact regions"""
        # Mock configuration
        self.lbw_predictor.config.stump_dimensions.width = 5.0
        self.lbw_predictor.config.stump_dimensions.height = 30.0
        self.lbw_predictor.config.stump_dimensions.bail_height = 2.0
        self.lbw_predictor.config.stump_dimensions.depth = 4.0
        self.lbw_predictor.config.pitch_dimensions.length = 40.0
        
        # Create mock confidence calculation
        with patch.object(self.lbw_predictor, '_calculate_confidence') as mock_confidence:
            mock_confidence.return_value = 0.7
            
            # Test low region impact
            low_path = [TrajectoryPoint(pos_x=2.5, pos_y=5.0, pos_z=20.0, timestamp=0.1)]
            hit, region, _ = self.lbw_predictor.check_stump_collision(low_path)
            self.assertTrue(hit)
            self.assertEqual(region, "low")
            
            # Test middle region impact
            middle_path = [TrajectoryPoint(pos_x=2.5, pos_y=15.0, pos_z=20.0, timestamp=0.1)]
            hit, region, _ = self.lbw_predictor.check_stump_collision(middle_path)
            self.assertTrue(hit)
            self.assertEqual(region, "middle")
            
            # Test high region impact
            high_path = [TrajectoryPoint(pos_x=2.5, pos_y=25.0, pos_z=20.0, timestamp=0.1)]
            hit, region, _ = self.lbw_predictor.check_stump_collision(high_path)
            self.assertTrue(hit)
            self.assertEqual(region, "high")
    
    def test_check_stump_collision_handles_grazing_impacts(self):
        """Test check_stump_collision correctly handles grazing impacts within tolerance"""
        # Mock configuration - set up stumps at exactly x=0-5, z=18-22
        self.lbw_predictor.config.stump_dimensions.width = 5.0
        self.lbw_predictor.config.stump_dimensions.height = 30.0
        self.lbw_predictor.config.stump_dimensions.bail_height = 2.0
        self.lbw_predictor.config.stump_dimensions.depth = 4.0
        self.lbw_predictor.config.pitch_dimensions.length = 40.0
        
        # Create a trajectory that just grazes the edge of stumps (within 1cm tolerance)
        grazing_path = [
            TrajectoryPoint(pos_x=5.0, pos_y=15.0, pos_z=20.0, timestamp=0.1)  # Exactly at edge
        ]
        
        # Mock confidence calculation
        with patch.object(self.lbw_predictor, '_calculate_confidence') as mock_confidence:
            mock_confidence.return_value = 0.6
            
            # Check collision
            hit, region, confidence = self.lbw_predictor.check_stump_collision(grazing_path)
            
            # Verify results
            self.assertTrue(hit)  # Should detect a hit
            self.assertEqual(region, "middle")
            self.assertEqual(confidence, 0.6)
