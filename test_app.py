import pytest
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from app import QuantumBioReactionMeter, get_ai_analysis

@pytest.fixture
def qbrm():
    return QuantumBioReactionMeter()

@pytest.fixture
def sample_data():
    # Create sample CSV data
    data = pd.DataFrame({
        'x_rotation_0': [0.5, 0.3],
        'x_rotation_1': [0.7, 0.4],
        'x_rotation_2': [0.2, 0.8],
        'x_rotation_3': [0.9, 0.1]
    })
    data.to_csv('test_data.csv', index=False)
    yield 'test_data.csv'
    # Cleanup
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')

def test_initialization():
    qbrm = QuantumBioReactionMeter()
    assert len(qbrm.organism_types) == 5
    assert 'Human' in qbrm.organism_types
    assert 'Lizard' in qbrm.organism_types
    assert 'Animal' in qbrm.organism_types
    assert 'Water Being' in qbrm.organism_types
    assert 'Plant' in qbrm.organism_types
    
    # Test gate configurations
    assert set(qbrm.organism_types['Human']['gates']) == {'X', 'H', 'CNOT'}
    assert set(qbrm.organism_types['Lizard']['gates']) == {'X', 'H'}
    assert qbrm.organism_types['Human']['qubits'] == 4

def test_circuit_creation(qbrm):
    # Test for each organism type
    for organism in qbrm.organism_types.keys():
        parameters = {f'x_rotation_{i}': 0.5 for i in range(qbrm.organism_types[organism]['qubits'])}
        circuit = qbrm.create_quantum_circuit(organism, parameters)
        assert circuit is not None
        assert len(circuit) > 0
        
        # Verify gate types
        gate_types = set(op.gate.__class__.__name__ for op in circuit.all_operations())
        if 'H' in qbrm.organism_types[organism]['gates']:
            assert 'HPowGate' in gate_types
        if 'X' in qbrm.organism_types[organism]['gates']:
            assert 'XPowGate' in gate_types
            if 'CNOT' in qbrm.organism_types[organism]['gates']:
                assert 'CXPowGate' in gate_types

def test_simulation(qbrm):
    # Test simulation for each organism type
    for organism, config in qbrm.organism_types.items():
        parameters = {f'x_rotation_{i}': 0.5 for i in range(config['qubits'])}
        circuit = qbrm.create_quantum_circuit(organism, parameters)
        state_vector = qbrm.simulate_reaction(circuit)
        
        assert state_vector is not None
        assert len(state_vector) == 2**config['qubits']
        assert isinstance(state_vector, np.ndarray)
        # Allow both complex64 and complex128
        assert state_vector.dtype in (np.complex64, np.complex128)

def test_results_processing(qbrm):
    # Test for each organism type
    for organism, config in qbrm.organism_types.items():
        parameters = {f'x_rotation_{i}': 0.5 for i in range(config['qubits'])}
        circuit = qbrm.create_quantum_circuit(organism, parameters)
        state_vector = qbrm.simulate_reaction(circuit)
        results = qbrm.process_results(state_vector, organism)
        
        assert 'probabilities' in results
        assert 'energy_levels' in results
        assert 'reaction_rates' in results
        
        # Check that probabilities sum to approximately 1
        assert abs(sum(results['probabilities']) - 1.0) < 1e-10
        
        # Check lengths
        expected_states = 2**config['qubits']
        assert len(results['probabilities']) == expected_states
        assert len(results['energy_levels']) == expected_states
        assert len(results['reaction_rates']) == expected_states

def test_csv_data_handling(qbrm, sample_data):
    # Test CSV data loading and processing
    data = pd.read_csv(sample_data)
    parameters = data.iloc[0].to_dict()
    
    # Test with Human (4 qubits)
    circuit = qbrm.create_quantum_circuit('Human', parameters)
    state_vector = qbrm.simulate_reaction(circuit)
    results = qbrm.process_results(state_vector, 'Human')
    
    assert all(key in parameters for key in [f'x_rotation_{i}' for i in range(4)])
    assert len(results['probabilities']) == 2**4

def test_result_file_generation(qbrm):
    # Test result file generation and format
    organism = 'Human'
    parameters = {f'x_rotation_{i}': 0.5 for i in range(4)}
    circuit = qbrm.create_quantum_circuit(organism, parameters)
    state_vector = qbrm.simulate_reaction(circuit)
    results = qbrm.process_results(state_vector, organism)
    
    # Save results
    results_data = {
        'organism_type': organism,
        'parameters': parameters,
        'results': results
    }
    
    filename = f'test_results_{organism}.json'
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=4)
    
    # Verify file exists and content
    assert os.path.exists(filename)
    with open(filename, 'r') as f:
        loaded_data = json.load(f)
    
    assert loaded_data['organism_type'] == organism
    assert loaded_data['parameters'] == parameters
    assert all(key in loaded_data['results'] for key in ['probabilities', 'energy_levels', 'reaction_rates'])
    
    # Cleanup
    os.remove(filename)

def test_ai_analysis():
    # Test AI analysis function with mock data
    test_data = {
        'organism_type': 'Human',
        'parameters': {'x_rotation_0': 0.5},
        'results': {
            'probabilities': [0.5, 0.5],
            'energy_levels': [-0.693, -0.693],
            'reaction_rates': [0.0, 0.0]
        }
    }
    
    # Test that the function runs without errors
    try:
        analysis = get_ai_analysis(test_data)
        assert isinstance(analysis, str)
    except Exception as e:
        # If API key is invalid or other API issues, function should handle gracefully
        assert "AI analysis currently unavailable" in str(e)

def test_invalid_parameters(qbrm):
    # Test various invalid parameter scenarios
    with pytest.raises(Exception):
        qbrm.create_quantum_circuit('NonexistentOrganism', {})
    
    # Test with missing required parameters
    with pytest.raises(KeyError):
        qbrm.create_quantum_circuit('Human', {'invalid_param': 0.5})
    
    # Test with out-of-range parameters
    invalid_params = {f'x_rotation_{i}': 2.0 for i in range(4)}  # Values should be between 0 and 1
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        qbrm.create_quantum_circuit('Human', invalid_params)

def test_result_types_and_ranges(qbrm):
    for organism, config in qbrm.organism_types.items():
        parameters = {f'x_rotation_{i}': 0.5 for i in range(config['qubits'])}
        circuit = qbrm.create_quantum_circuit(organism, parameters)
        state_vector = qbrm.simulate_reaction(circuit)
        results = qbrm.process_results(state_vector, organism)
        
        # Check types
        assert isinstance(results['probabilities'], list)
        assert isinstance(results['energy_levels'], list)
        assert isinstance(results['reaction_rates'], list)
        
        # Check value ranges
        assert all(0 <= p <= 1 for p in results['probabilities'])
        assert all(isinstance(e, float) for e in results['energy_levels'])
        assert all(isinstance(r, float) for r in results['reaction_rates'])
