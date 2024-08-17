import React from 'react';
import './App.css';
import RnnPredictor from './RnnPredictor';

function App() {
  return (
    <div className='app'>
      <h1>Number predictor</h1>
      <p>This is a number predictor. You choose the model arcitecture you would like, the unit number and number of epochs for training. Next start picking numbers 0-9. Then the model will try to predict the next number you would choose. Have fun!</p>
      <RnnPredictor />
    </div>
  );
}

export default App;
