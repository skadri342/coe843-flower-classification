import { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import PredictionResult from './components/PredictionResult';

const App = () => {
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredictionStart = () => {
    setIsLoading(true);
    setPrediction(null);
  };

  const handlePredictionComplete = (result) => {
    setPrediction(result);
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h1 className="text-2xl font-bold mb-4">Flower Classification</h1>
          <div className="space-y-4">
            <ImageUpload 
              onPredictionStart={handlePredictionStart}
              onPredictionComplete={handlePredictionComplete}
            />
            {(prediction || isLoading) && (
              <PredictionResult 
                prediction={prediction}
                isLoading={isLoading}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;