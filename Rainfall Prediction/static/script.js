let modelResults = null;
let charts = {};

// Load model results when the page loads
fetch('/model_results')
    .then(response => response.json())
    .then(data => {
        console.log('Model results loaded:', data); // Debug log
        modelResults = data;
        updateCharts();
    })
    .catch(error => {
        console.error('Error loading model results:', error);
        alert('Error loading model results. Please check the console for details.');
    });

function updateCharts() {
    if (!modelResults) {
        console.error('No model results available');
        return;
    }

    const selectedModel = document.getElementById('model').value;
    console.log('Updating charts for model:', selectedModel); // Debug log

    try {
        // Accuracy Chart
        const accuracyCtx = document.getElementById('accuracyChart');
        if (!accuracyCtx) {
            console.error('Accuracy chart canvas not found');
            return;
        }
        if (charts.accuracy) charts.accuracy.destroy();
        charts.accuracy = new Chart(accuracyCtx, {
            type: 'bar',
            data: {
                labels: Object.keys(modelResults),
                datasets: [{
                    label: 'Model Accuracy',
                    data: Object.values(modelResults).map(r => r.accuracy),
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Accuracy Comparison'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });

        // ROC Curve Chart
        const rocCtx = document.getElementById('rocCurveChart');
        if (rocCtx) {
            if (charts.roc) charts.roc.destroy();
            charts.roc = new Chart(rocCtx, {
                type: 'line',
                data: {
                    labels: modelResults[selectedModel].roc_curve.fpr,
                    datasets: [{
                        label: `ROC Curve (AUC = ${modelResults[selectedModel].roc_curve.auc.toFixed(3)})`,
                        data: modelResults[selectedModel].roc_curve.tpr,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: `ROC Curve - ${selectedModel}`
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'False Positive Rate'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'True Positive Rate'
                            }
                        }
                    }
                }
            });
        }

        // Learning Curve Chart
        const learningCtx = document.getElementById('learningCurveChart');
        if (learningCtx) {
            if (charts.learning) charts.learning.destroy();
            charts.learning = new Chart(learningCtx, {
                type: 'line',
                data: {
                    labels: modelResults[selectedModel].learning_curve.train_sizes,
                    datasets: [
                        {
                            label: 'Training Score',
                            data: modelResults[selectedModel].learning_curve.train_scores_mean,
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            fill: true
                        },
                        {
                            label: 'Cross-validation Score',
                            data: modelResults[selectedModel].learning_curve.test_scores_mean,
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: `Learning Curve - ${selectedModel}`
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Training Examples'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Score'
                            }
                        }
                    }
                }
            });
        }

        // Feature Importance Chart
        if (modelResults[selectedModel].feature_importance) {
            const featureCtx = document.getElementById('featureImportanceChart');
            if (featureCtx) {
                if (charts.feature) charts.feature.destroy();
                charts.feature = new Chart(featureCtx, {
                    type: 'bar',
                    data: {
                        labels: ['Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Wind Direction'],
                        datasets: [{
                            label: 'Feature Importance',
                            data: modelResults[selectedModel].feature_importance,
                            backgroundColor: 'rgba(153, 102, 255, 0.5)',
                            borderColor: 'rgba(153, 102, 255, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: `Feature Importance - ${selectedModel}`
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            }
        }

        // Confusion Matrix Chart
        const confusionCtx = document.getElementById('confusionMatrixChart');
        if (confusionCtx) {
            if (charts.confusion) charts.confusion.destroy();
            charts.confusion = new Chart(confusionCtx, {
                type: 'bar',
                data: {
                    labels: ['True Negative', 'False Positive', 'False Negative', 'True Positive'],
                    datasets: [{
                        label: 'Confusion Matrix',
                        data: modelResults[selectedModel].confusion_matrix.flat(),
                        backgroundColor: [
                            'rgba(75, 192, 192, 0.5)',
                            'rgba(255, 99, 132, 0.5)',
                            'rgba(255, 99, 132, 0.5)',
                            'rgba(75, 192, 192, 0.5)'
                        ],
                        borderColor: [
                            'rgba(75, 192, 192, 1)',
                            'rgba(255, 99, 132, 1)',
                            'rgba(255, 99, 132, 1)',
                            'rgba(75, 192, 192, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: `Confusion Matrix - ${selectedModel}`
                        }
                    }
                }
            });
        }
    } catch (error) {
        console.error('Error updating charts:', error);
        alert('Error updating charts. Please check the console for details.');
    }
}

// Update charts when model selection changes
document.getElementById('model').addEventListener('change', updateCharts);

// Handle form submission
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Get form data
    const formData = {
        model: document.getElementById('model').value,
        temperature: document.getElementById('temperature').value,
        humidity: document.getElementById('humidity').value,
        pressure: document.getElementById('pressure').value,
        wind_speed: document.getElementById('wind_speed').value,
        wind_direction: document.getElementById('wind_direction').value
    };

    try {
        // Send prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        // Display results
        const resultDiv = document.getElementById('result');
        const predictionText = document.getElementById('predictionText');
        const probabilityText = document.getElementById('probabilityText');
        const modelUsed = document.getElementById('modelUsed');

        predictionText.textContent = data.prediction;
        probabilityText.textContent = `Probability: ${data.probability}`;
        modelUsed.textContent = `Model used: ${data.model_used}`;
        resultDiv.style.display = 'block';

        // Add animation
        resultDiv.style.animation = 'none';
        resultDiv.offsetHeight; // Trigger reflow
        resultDiv.style.animation = 'fadeIn 0.5s ease-in-out';

    } catch (error) {
        alert('Error occurred while making prediction. Please try again.');
        console.error('Error:', error);
    }
}); 