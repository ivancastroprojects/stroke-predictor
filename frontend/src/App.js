import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainComponent from './components/MainComponent';
import PredictionResults from './components/PredictionResults';
import BackgroundRays from './components/BackgroundRays';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <BackgroundRays />
        <Routes>
          <Route path="/" element={<MainComponent />} />
          <Route path="/results" element={<PredictionResults />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
