import React, { useState, useEffect, ChangeEvent } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Bar, Line } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

const RnnPredictor: React.FC = () => {
  const [sequence, setSequence] = useState<string>('');
  const [prediction, setPrediction] = useState<number | null>(null);
  const [probabilities, setProbabilities] = useState<number[]>([])
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [epochs, setEpochs] = useState(5);
  const [units, setUnits] = useState(10);
  const [modelType, setModelType] = useState('LSTM'); // Model type selection
  const [lossData, setLossData] = useState<number[]>([]);
  const [labels, setLabels] = useState<number[]>([]);
  const [isTraining, setIsTraining] = useState(false); // State to track training status

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
      setIsTraining(true); // Set training status to true
      const seqArray = sequence.split('').map(Number);
      const inputTensor = tf.tensor(seqArray.slice(0, -1)).reshape([seqArray.length - 1, 1, 1]);
      const targetTensor = tf.tensor(seqArray.slice(1)).reshape([seqArray.length - 1, 1]);

      model.fit(inputTensor, targetTensor, {
        epochs,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            setLossData(prev => [...prev, logs?.loss ?? 0]);
            setLabels(prev => [...prev, epoch + 1]);
          },
          onTrainEnd: () => {
            setIsTraining(false); // Set training status to false after training is done
          },
        },
      }).then(() => {
        const lastInput = tf.tensor([seqArray[seqArray.length - 1]]).reshape([1, 1, 1]);
        const output = model.predict(lastInput) as tf.Tensor;
        setPrediction(tf.argMax(output, 1).dataSync()[0]);
        setProbabilities(Array.from(output.dataSync()));
      });
    }
  }, [sequence, model, epochs]);

  // Handle user input
  const handleClick = (num: number) => {
    setSequence(prev => prev + num);
    setLossData([]); // Reset loss data for new training
    setLabels([]);
  };

  // Handle direct editing of the sequence
  const handleSequenceChange = (e: ChangeEvent<HTMLInputElement>) => {
    const input = e.target.value.replace(/[^0-9]/g, ''); // Allow only digits
    setSequence(input);
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
        <select className='models' value={modelType} onChange={handleModelTypeChange}>
          <option value="SimpleRNN">Simple RNN</option>
          <option value="LSTM">LSTM</option>
          <option value="GRU">GRU</option>
        </select>
      </div>
      <div>
        <label>Units: </label>
        <input className='units' type="number" value={units} onChange={handleUnitsChange} min="1" max="100" />
      </div>
      <div>
        <label>Epochs: </label>
        <input className='epochs' type="number" value={epochs} onChange={handleEpochsChange} min="1" max="100" />
      </div>
      <h3>Pick your numbers:</h3>
      <div className='buttonWrapper'>
        {Array.from({ length: 10 }, (_, i) => (
          <button className='button' key={i} onClick={() => handleClick(i)} disabled={isTraining}>{i}</button>
        ))}
      </div>
      <div>
        <label>Sequence: </label>
        <input className='sequence' type="text" value={sequence.split('')} onChange={handleSequenceChange} disabled={isTraining} />
      </div>

      {isTraining && <div>Training in progress... <span className="spinner"></span></div>}
      <div className='prediction'>Prediction: {prediction !== null ? prediction : 'N/A'}</div>
      <div className='disclaimer'>Disclaimer: this is a small dataset on a small model, do not expect much...</div>


      <div>
        <h3>Output Probabilities</h3>
        <Bar
          className='plot'
          data={{
            labels: Array.from({ length: 10 }, (_, i) => i.toString()),
            datasets: [{
              label: 'Probability',
              data: probabilities,
              backgroundColor: 'rgba(54, 162, 235, 0.2)',
              borderColor: 'rgba(54, 162, 235, 1)',
              borderWidth: 1,
            }]
          }}
          options={{
            scales: {
              y: {
                beginAtZero: true,
                max: 1,
              }
            }
          }}
        />
      </div>
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
      <p>wow! You scrolled all the way down? The fun is up there ^.</p>
      <style>{`
        .spinner {
          border: 4px solid rgba(0, 0, 0, 0.1);
          border-left-color: #4caf50;
          border-radius: 50%;
          width: 24px;
          height: 24px;
          animation: spin 1s linear infinite;
          display: inline-block;
          margin-left: 10px;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default RnnPredictor;
