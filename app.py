import streamlit as st
import cirq
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client if API key is available
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

def get_ai_analysis(data_dict):
    """Get AI analysis of the quantum bio-reaction data"""
    if not client:
        return """AI analysis is not available. To enable AI analysis:
1. Create a .env file in the project root
2. Add your OpenAI API key: OPENAI_API_KEY=your_key_here
3. Restart the application"""
        
    prompt = f"""Analyze this quantum bio-reaction data and provide detailed insights:
    Organism Type: {data_dict['organism_type']}
    Parameters: {data_dict['parameters']}
    Probabilities: {data_dict['results']['probabilities'][:5]}... (first 5 values)
    Energy Levels: {data_dict['results']['energy_levels'][:5]}... (first 5 values)
    Reaction Rates: {data_dict['results']['reaction_rates'][:5]}... (first 5 values)
    
    Please provide:
    1. A detailed analysis of the quantum state
    2. Biological implications
    3. Potential applications
    4. Recommendations for further investigation
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a quantum biology expert analyzing bio-reaction data from quantum simulations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in AI analysis: {str(e)}")
        return "AI analysis currently unavailable. Please try again later."

class QuantumBioReactionMeter:
    def __init__(self):
        self.organism_types = {
            'Human': {'qubits': 4, 'gates': ['X', 'H', 'CNOT']},
            'Lizard': {'qubits': 3, 'gates': ['X', 'H']},
            'Animal': {'qubits': 4, 'gates': ['X', 'H', 'CNOT']},
            'Water Being': {'qubits': 2, 'gates': ['H']},
            'Plant': {'qubits': 2, 'gates': ['X']}
        }
        
    def create_quantum_circuit(self, organism_type, parameters):
        if organism_type not in self.organism_types:
            raise Exception(f"Unknown organism type: {organism_type}")
            
        num_qubits = self.organism_types[organism_type]['qubits']
        qubits = cirq.LineQubit.range(num_qubits)
        circuit = cirq.Circuit()
        
        # Validate parameters
        required_params = [f'x_rotation_{i}' for i in range(num_qubits)]
        for param in required_params:
            if param not in parameters:
                raise KeyError(f"Missing required parameter: {param}")
            if not 0 <= parameters[param] <= 1:
                raise ValueError(f"Parameter {param} must be between 0 and 1")
        
        # Apply quantum gates based on organism type
        for i, qubit in enumerate(qubits):
            if 'H' in self.organism_types[organism_type]['gates']:
                circuit.append(cirq.H(qubit))
            if 'X' in self.organism_types[organism_type]['gates']:
                circuit.append(cirq.X(qubit)**parameters[f'x_rotation_{i}'])
                
        # Add CNOT gates for complex organisms
        if 'CNOT' in self.organism_types[organism_type]['gates']:
            for i in range(num_qubits - 1):
                circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
                
        return circuit
    
    def simulate_reaction(self, circuit):
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        return result.final_state_vector
    
    def process_results(self, state_vector, organism_type):
        # Convert quantum state to reaction metrics
        probabilities = np.abs(state_vector)**2
        # Normalize probabilities to ensure they sum to 1
        probabilities = probabilities / np.sum(probabilities)
        energy_levels = np.real(np.log(probabilities + 1e-10))
        reaction_rates = np.imag(state_vector)
        
        return {
            'probabilities': probabilities.tolist(),
            'energy_levels': energy_levels.tolist(),
            'reaction_rates': reaction_rates.tolist()
        }

def main():
    st.set_page_config(page_title="Quantum AI BioReaction Meter", layout="wide")
    st.title("Quantum AI BioReaction Meter")
    
    # Add description
    st.markdown("""
    This advanced tool combines quantum computing with AI to analyze biological reactions at the quantum level.
    Select an organism type and configure parameters to simulate quantum bio-reactions.
    """)
    
    qbrm = QuantumBioReactionMeter()
    
    # Sidebar for input method selection
    input_method = st.sidebar.radio("Select Input Method", ["CSV Upload", "Manual Configuration"])
    
    # Organism selection
    organism_type = st.sidebar.selectbox(
        "Select Organism Type",
        list(qbrm.organism_types.keys())
    )
    
    # Initialize parameters with default values
    num_qubits = qbrm.organism_types[organism_type]['qubits']
    parameters = {f'x_rotation_{i}': 0.5 for i in range(num_qubits)}
    
    if input_method == "CSV Upload":
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:", data.head())
            csv_parameters = data.iloc[0].to_dict()
            # Update only valid parameters from CSV
            for key in parameters.keys():
                if key in csv_parameters:
                    parameters[key] = csv_parameters[key]
    else:
        # Manual configuration
        st.sidebar.subheader("Parameter Configuration")
        for i in range(num_qubits):
            parameters[f'x_rotation_{i}'] = st.sidebar.slider(
                f"X Rotation {i+1}",
                0.0, 1.0, parameters[f'x_rotation_{i}'],
                step=0.1
            )
    
    if st.button("Run Simulation"):
        with st.spinner("Running quantum simulation..."):
            # Create and run quantum circuit
            circuit = qbrm.create_quantum_circuit(organism_type, parameters)
            state_vector = qbrm.simulate_reaction(circuit)
            results = qbrm.process_results(state_vector, organism_type)
            
            # Create tabs for different visualizations
            tabs = st.tabs(["Circuit", "Basic Analysis", "Advanced Visualizations", "AI Analysis"])
            
            with tabs[0]:
                st.subheader("Quantum Circuit")
                st.text(str(circuit))
                
                # Circuit statistics
                st.info(f"""
                Circuit Statistics:
                - Number of qubits: {len(circuit.all_qubits())}
                - Number of operations: {len(list(circuit.all_operations()))}
                - Gate types: {set(op.gate.__class__.__name__ for op in circuit.all_operations())}
                """)
            
            with tabs[1]:
                col1, col2 = st.columns(2)
            
            with col1:
                # Energy levels plot
                fig_energy = px.line(
                    y=results['energy_levels'],
                    title="Energy Levels",
                    labels={'index': 'State', 'value': 'Energy'}
                )
                st.plotly_chart(fig_energy)
            
            with col2:
                # Reaction rates plot
                fig_rates = px.bar(
                    y=results['reaction_rates'],
                    title="Reaction Rates",
                    labels={'index': 'State', 'value': 'Rate'}
                )
                st.plotly_chart(fig_rates)
            
            with tabs[2]:
                # Enhanced 3D visualization
                fig_3d = go.Figure()
                
                # Add quantum state trajectory
                fig_3d.add_trace(go.Scatter3d(
                    x=np.real(state_vector),
                    y=np.imag(state_vector),
                    z=results['probabilities'],
                    mode='markers+lines',
                    marker=dict(
                        size=10,
                        color=results['energy_levels'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    line=dict(color='rgba(100,100,100,0.3)'),
                    name='Quantum State'
                ))
                
                # Add probability distribution
                fig_3d.add_trace(go.Scatter3d(
                    x=np.real(state_vector),
                    y=[0]*len(state_vector),
                    z=results['probabilities'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=results['probabilities'],
                        colorscale='Plasma',
                        showscale=True
                    ),
                    name='Probability Distribution'
                ))
                fig_3d.update_layout(
                    title="Interactive Quantum State Visualization",
                    scene=dict(
                        xaxis_title="Real Component",
                        yaxis_title="Imaginary Component",
                        zaxis_title="Probability",
                        camera=dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )
                    ),
                    showlegend=True,
                    width=800,
                    height=600
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Add phase space plot
                phase_space = go.Figure()
                phase_space.add_trace(go.Scatter(
                    x=np.real(state_vector),
                    y=np.imag(state_vector),
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=results['probabilities'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f'State {i}' for i in range(len(state_vector))],
                    textposition="top center",
                    name='Quantum States'
                ))
                phase_space.update_layout(
                    title="Phase Space Representation",
                    xaxis_title="Real Component",
                    yaxis_title="Imaginary Component",
                    width=800,
                    height=500
                )
                st.plotly_chart(phase_space, use_container_width=True)
            
            with tabs[3]:
                st.subheader("AI Analysis")
                
                # Get AI analysis
                data_for_analysis = {
                    'organism_type': organism_type,
                    'parameters': parameters,
                    'results': results
                }
                
                with st.spinner("Generating AI analysis..."):
                    ai_analysis = get_ai_analysis(data_for_analysis)
                    st.markdown(ai_analysis)
            
            # Enhanced results saving
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"quantum_ai_results_{organism_type}_{timestamp}.json"
            
            with open(results_filename, 'w') as f:
                json.dump({
                    'organism_type': organism_type,
                    'parameters': parameters,
                    'results': results
                }, f, indent=4)
            
            # Download button
            with open(results_filename, 'rb') as f:
                st.download_button(
                    label="Download Results",
                    data=f,
                    file_name=results_filename,
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
