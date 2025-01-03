# Quantum AI BioReaction Meter

A sophisticated tool that combines quantum computing with AI to analyze biological reactions at the quantum level. This application simulates and visualizes quantum bio-reactions for various organisms using Cirq for quantum computations and OpenAI for advanced analysis.

## Features

- **Multiple Organism Support**: Analyze quantum reactions for:
  - Humans (4 qubits)
  - Lizards (3 qubits)
  - Animals (4 qubits)
  - Water Beings (2 qubits)
  - Plants (2 qubits)

- **Interactive Interface**:
  - CSV data upload support
  - Manual parameter configuration
  - Real-time visualization
  - AI-powered analysis

- **Visualizations**:
  - Quantum circuit diagrams
  - Energy level plots
  - Reaction rate analysis
  - 3D quantum state visualization
  - Phase space representation

- **Data Export**:
  - JSON format results
  - Downloadable analysis reports
  - Parameter configurations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quantum-bio-reaction-meter.git
cd quantum-bio-reaction-meter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Select an organism type from the sidebar

3. Choose input method:
   - Upload CSV file with parameters
   - Use manual configuration sliders

4. Click "Run Simulation" to generate results

## CSV Format

Sample CSV format for parameter input:
```csv
x_rotation_0,x_rotation_1,x_rotation_2,x_rotation_3
0.5,0.7,0.3,0.9
```

## Testing

Run the test suite:
```bash
pytest test_app.py -v
```

## Technical Details

### Quantum Circuit Components

- **Gates Used**:
  - Hadamard (H): Creates superposition
  - X Gate: Implements rotations
  - CNOT: Entangles qubits

### Analysis Metrics

- Quantum state probabilities
- Energy level distributions
- Reaction rates
- Phase space representations

### AI Integration

The system uses GPT-4 to provide:
- Detailed quantum state analysis
- Biological implications
- Potential applications
- Research recommendations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Cirq](https://quantumai.google/cirq) quantum computing framework
- AI analysis powered by [OpenAI](https://openai.com/)
- Interactive interface using [Streamlit](https://streamlit.io/)
