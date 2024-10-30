const PredictionResult = ({ prediction, isLoading }) => {
  if (isLoading) {
    return (
      <div className="mt-4 text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 mx-auto"></div>
        <p className="mt-2 text-gray-600">Analyzing image...</p>
      </div>
    );
  }

  if (prediction?.error) {
    return (
      <div className="mt-4 p-4 bg-red-100 text-red-700 rounded">
        {prediction.error}
      </div>
    );
  }

  if (!prediction) return null;

  return (
    <div className="mt-4 p-4 bg-green-100 rounded">
      <h3 className="text-lg font-semibold">Results:</h3>
      <p className="mt-2">Flower Type: {prediction.prediction}</p>
      <p>Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
    </div>
  );
};

export default PredictionResult;