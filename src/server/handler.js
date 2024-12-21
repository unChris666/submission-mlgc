const predictClassification = require("../services/InferenceService");
const crypto = require("crypto");
const storeData = require("../services/storeData");

async function postPredictHandler(request, h) {
  const { image } = request.payload;
  const { model } = request.server.app;

// Check if image is provided
if (!image) {
  return h.response({
      status: 'fail',
      message: 'Image is required'
  }).code(400); // Bad Request
}

const { confidenceScore, result, suggestion } = await predictClassification(model, image);

  const id = crypto.randomUUID();
  const createdAt = new Date().toISOString();

  const data = {
    "id": id,
    "result": result,
    "suggestion": suggestion,
    "createdAt": createdAt,
  };

  await storeData(id, data);

  const response = h.response({
    status: "success",
    message:
      confidenceScore > 99
        ? "Model is predicted successfully"
        : "Model is predicted successfully",
    data,
  });
  response.code(201);
  return response;
}

module.exports = { postPredictHandler };