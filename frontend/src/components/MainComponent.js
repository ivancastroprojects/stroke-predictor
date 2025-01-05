import React, { useState, useEffect } from 'react';
import PredictionHistory from './PredictionHistory';
import PredictionResults from './PredictionResults';

export default function MainComponent() {
  const [userPredictions, setUserPredictions] = useState([]);
  const [predictions, setPredictions] = useState(examplePredictions);

  const handleNewPrediction = (newPrediction) => {
    setUserPredictions(prev => [...prev, newPrediction]);
  };

  useEffect(() => {
    if (userPredictions.length > 0) {
      setPredictions(userPredictions);
    }
  }, [userPredictions]);

  return (
    <div>
      <PredictionResults onNewPrediction={handleNewPrediction} />
      <PredictionHistory userPredictions={predictions} />
    </div>
  );
} 