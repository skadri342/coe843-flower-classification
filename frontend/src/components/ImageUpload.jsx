import { useState } from 'react';
import axios from 'axios';
import PropTypes from 'prop-types';

const API_URL = import.meta.env.VITE_API_URL;

const ImageUpload = ({ onPredictionStart, onPredictionComplete }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select an image file');
        return;
      }
      setSelectedFile(file);
      setError(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      onPredictionStart();
      const response = await axios.post(`${API_URL}/mlapi/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      onPredictionComplete(response.data);
    } catch (error) {
      console.error('Error:', error);
      onPredictionComplete({ error: error.response?.data?.error || 'Failed to process image' });
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-center w-full">
        <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
          {previewUrl ? (
            <div className="relative w-full h-full">
              <img 
                src={previewUrl} 
                alt="Preview" 
                className="absolute inset-0 w-full h-full object-contain p-2"
              />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <svg className="w-8 h-8 mb-4 text-gray-500" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
              </svg>
              <p className="mb-2 text-sm text-gray-500"><span className="font-semibold">Click to upload</span> or drag and drop</p>
              <p className="text-xs text-gray-500">PNG, JPG or JPEG</p>
            </div>
          )}
          <input 
            type="file" 
            className="hidden" 
            onChange={handleFileSelect}
            accept="image/*"
          />
        </label>
      </div>
      {error && (
        <p className="text-red-500 text-center">{error}</p>
      )}
      {selectedFile && (
        <div className="text-center">
          <p className="text-sm text-gray-500">Selected: {selectedFile.name}</p>
          <button onClick={handleSubmit} className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors">
            Classify Image
          </button>
        </div>
      )}
    </div>
  );
};

// PropTypes validation
ImageUpload.propTypes = {
  onPredictionStart: PropTypes.func.isRequired,
  onPredictionComplete: PropTypes.func.isRequired
};

export default ImageUpload;