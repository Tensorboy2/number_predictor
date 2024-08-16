import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Line } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

const RnnPredictor: React.FC = () => {
  const [sequence, setSequence] = useState<number[]>([]);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [epochs, setEpochs] = useState(5);
  const [units, setUnits] = useState(10);
  const [modelType, setModelType] = useState('LSTM'); // Model type selection
  const [lossData, setLossData] = useState<number[]>([]);
  const [labels, setLabels] = useState<number[]>([]);

  // Initialize model based on user selection
  useEffect(() => {
    const buildModel = () => {
      const model = tf.sequential();
      switch (modelType) {
        case 'LSTM':
          model.add(tf.layers.lstm({ units, inputShape: [1, 1] }));
          break;
        case 'GRU':
          model.add(tf.layers.gru({ units, inputShape: [1, 1] }));
          break;
        default: // SimpleRNN
          model.add(tf.layers.simpleRNN({ units, inputShape: [1, 1] }));
          break;
      }
      model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
      model.compile({ loss: 'sparseCategoricalCrossentropy', optimizer: 'adam' });
      return model;
    };

    setModel(buildModel());
  }, [units, modelType]);

  // Train the model whenever sequence changes
  useEffect(() => {
    if (model && sequence.length > 1) {
      const inputTensor = tf.tensor(sequence.slice(0, -1)).reshape([sequence.length - 1, 1, 1]);
      const targetTensor = tf.tensor(sequence.slice(1)).reshape([sequence.length - 1, 1]);
      
      model.fit(inputTensor, targetTensor, {
        epochs,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            setLossData(prev => [...prev, logs?.loss ?? 0]);
            setLabels(prev => [...prev, epoch + 1]);
          }
        }
      }).then(() => {
        const lastInput = tf.tensor([sequence[sequence.length - 1]]).reshape([1, 1, 1]);
        const output = model.predict(lastInput) as tf.Tensor;
        setPrediction(tf.argMax(output, 1).dataSync()[0]);
      });
    }
  }, [sequence, model, epochs]);

  // Handle user input
  const handleClick = (num: number) => {
    setSequence([...sequence, num]);
    setLossData([]); // Reset loss data for new training
    setLabels([]);
  };

  // Handle hyperparameter and model type changes
  const handleUnitsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUnits(Number(e.target.value));
  };

  const handleEpochsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEpochs(Number(e.target.value));
  };

  const handleModelTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setModelType(e.target.value);
  };

  return (
    <div>
      <div>
        <label>Model Type: </label>
        <select value={modelType} onChange={handleModelTypeChange}>
          <option value="SimpleRNN">Simple RNN</option>
          <option value="LSTM">LSTM</option>
          <option value="GRU">GRU</option>
        </select>
      </div>
      <div>
        <label>Units: </label>
        <input type="number" value={units} onChange={handleUnitsChange} min="1" max="100" />
      </div>
      <div>
        <label>Epochs: </label>
        <input type="number" value={epochs} onChange={handleEpochsChange} min="1" max="100" />
      </div>
      <div className='buttonWrapper'>
        {Array.from({ length: 10 }, (_, i) => (
          <button className='button' key={i} onClick={() => handleClick(i)}>{i}</button>
        ))}
      </div>
      <div>Sequence: {sequence.join(', ')}</div>
      <div className='prediction'>Prediction: {prediction !== null ? prediction : 'N/A'}</div>

      <div>
        <h3>Training Loss Over Epochs</h3>
        <Line
          className='plot'
          data={{
            labels: labels,
            datasets: [{
              label: 'Loss',
              data: lossData,
              fill: false,
              borderColor: 'rgba(75, 192, 192, 1)',
              tension: 0.1,
            }]
          }}
        />
      </div>
    </div>
  );
};

export default RnnPredictor;
