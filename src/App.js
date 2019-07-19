import React from 'react';
import './App.css';
// import Classifier from './classifier';
import KnnClassifier from './KnnClassifier';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        {/* <Classifier /> */}
        <KnnClassifier />
      </header>
    </div>
  );
}

export default App;
