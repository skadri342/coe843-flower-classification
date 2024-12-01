const PredictionResult = ({ prediction, isLoading }) => {
  // Map flower types to facts
  const flowerFacts = {
    daisy: "Daisies symbolize purity and innocence.",
    dandelion: "Dandelions are known for their medicinal properties and are often used in teas.",
    rose: "Roses are a symbol of love and passion, with thousands of varieties worldwide.",
    sunflower: "Sunflowers can grow up to 12 feet tall and are known for following the sun.",
    tulip: "Tulips were so valuable in the 1600s that they caused 'Tulip Mania' in the Netherlands."
  };

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

  const flowerType = prediction.prediction;
  const flowerFact = flowerFacts[flowerType] || "No fact available for this flower.";

  return (
    <div className="mt-4 p-4 bg-green-100 rounded">
      <h3 className="text-lg font-semibold">Results:</h3>
      <p className="mt-2">Flower Type: {flowerType}</p>
      <p>Confidence: {(prediction.confidence).toFixed(1)}%</p>
      <p className="mt-2 italic">{flowerFact}</p>
    </div>
  );
};

export default PredictionResult;