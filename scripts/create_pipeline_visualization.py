#!/usr/bin/env python3
"""
Pipeline Visualization - HTML Generator
Creates an interactive, animated HTML visualization of the project pipeline
"""

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Visualization</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }

        .header {
            text-align: center;
            margin-bottom: 60px;
        }

        .header h1 {
            font-size: 2.5em;
            color: #2C3E50;
            margin-bottom: 10px;
            animation: fadeInDown 1s ease;
        }

        .header p {
            color: #7F8C8D;
            font-size: 1.2em;
            animation: fadeInUp 1s ease;
        }

        /* Pipeline Container */
        .pipeline {
            position: relative;
            padding: 40px 20px;
        }

        /* Main Pipeline Tube */
        .pipeline-tube {
            position: absolute;
            top: 50%;
            left: 5%;
            right: 5%;
            height: 8px;
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #95E1D3, #F38181, #AA96DA);
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            z-index: 1;
            animation: flowGradient 3s ease-in-out infinite;
        }

        @keyframes flowGradient {
            0%, 100% {
                opacity: 0.8;
            }
            50% {
                opacity: 1;
                box-shadow: 0 4px 25px rgba(102, 126, 234, 0.5);
            }
        }

        /* Flow particles */
        .pipeline-tube::before {
            content: '';
            position: absolute;
            top: -2px;
            left: 0;
            width: 100%;
            height: 12px;
            background: linear-gradient(90deg, 
                transparent 0%, 
                rgba(255,255,255,0.8) 50%, 
                transparent 100%);
            border-radius: 10px;
            animation: flowParticles 2s linear infinite;
        }

        @keyframes flowParticles {
            0% {
                transform: translateX(-100%);
            }
            100% {
                transform: translateX(100%);
            }
        }

        /* Stages Container */
        .stages {
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: relative;
            z-index: 2;
            gap: 20px;
        }

        .stage {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            animation: fadeInScale 0.8s ease backwards;
        }

        .stage:nth-child(1) { animation-delay: 0.2s; }
        .stage:nth-child(2) { animation-delay: 0.4s; }
        .stage:nth-child(3) { animation-delay: 0.6s; }
        .stage:nth-child(4) { animation-delay: 0.8s; }
        .stage:nth-child(5) { animation-delay: 1s; }

        /* Pipeline Node (Circle) */
        .node {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5em;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            position: relative;
            cursor: pointer;
            transition: all 0.3s ease;
            animation: pulse 3s ease-in-out infinite;
            z-index: 3;
        }

        .node:hover {
            transform: scale(1.15) translateY(-5px);
            box-shadow: 0 12px 35px rgba(0,0,0,0.3);
        }

        .stage:nth-child(1) .node {
            background: linear-gradient(135deg, #FF6B6B, #EE5A6F);
            animation-delay: 0s;
        }
        .stage:nth-child(2) .node {
            background: linear-gradient(135deg, #4ECDC4, #44A08D);
            animation-delay: 0.6s;
        }
        .stage:nth-child(3) .node {
            background: linear-gradient(135deg, #95E1D3, #80CBC4);
            animation-delay: 1.2s;
        }
        .stage:nth-child(4) .node {
            background: linear-gradient(135deg, #F38181, #E66767);
            animation-delay: 1.8s;
        }
        .stage:nth-child(5) .node {
            background: linear-gradient(135deg, #AA96DA, #9575CD);
            animation-delay: 2.4s;
        }

        /* Ripple effect on nodes */
        .node::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            border: 3px solid currentColor;
            transform: translate(-50%, -50%);
            animation: ripple 2s ease-out infinite;
            opacity: 0;
        }

        @keyframes ripple {
            0% {
                width: 100%;
                height: 100%;
                opacity: 0.6;
            }
            100% {
                width: 160%;
                height: 160%;
                opacity: 0;
            }
        }

        @keyframes pulse {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
        }

        /* Stage Label */
        .stage-label {
            margin-top: 20px;
            font-weight: bold;
            font-size: 0.9em;
            color: #2C3E50;
            text-align: center;
            padding: 8px 16px;
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            white-space: nowrap;
        }

        /* Details Container */
        .details {
            margin-top: 15px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            width: 100%;
            max-width: 200px;
        }

        .detail-item {
            background: white;
            padding: 10px 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            font-size: 0.85em;
            color: #2C3E50;
            text-align: center;
            transition: all 0.3s ease;
            border-left: 3px solid;
        }

        .detail-item:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }

        .stage:nth-child(1) .detail-item { border-color: #FF6B6B; }
        .stage:nth-child(2) .detail-item { border-color: #4ECDC4; }
        .stage:nth-child(3) .detail-item { border-color: #95E1D3; }
        .stage:nth-child(4) .detail-item { border-color: #F38181; }
        .stage:nth-child(5) .detail-item { border-color: #AA96DA; }

        .detail-title {
            font-weight: bold;
            margin-bottom: 3px;
        }

        .detail-subtitle {
            font-size: 0.9em;
            color: #7F8C8D;
        }

        /* Stats Dashboard */
        .stats {
            margin-top: 60px;
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }

        .stat-card {
            padding: 20px 30px;
            border-radius: 15px;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }

        .stat-card:nth-child(1) { background: linear-gradient(135deg, #FF6B6B, #EE5A6F); }
        .stat-card:nth-child(2) { background: linear-gradient(135deg, #4ECDC4, #44A08D); }
        .stat-card:nth-child(3) { background: linear-gradient(135deg, #95E1D3, #80CBC4); }
        .stat-card:nth-child(4) { background: linear-gradient(135deg, #F38181, #E66767); }
        .stat-card:nth-child(5) { background: linear-gradient(135deg, #AA96DA, #9575CD); }

        .stat-label {
            font-size: 0.8em;
            opacity: 0.9;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
        }

        .info {
            margin-top: 30px;
            text-align: center;
            color: #7F8C8D;
            font-size: 0.9em;
        }

        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInScale {
            from {
                opacity: 0;
                transform: scale(0.7);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }

        @media (max-width: 1200px) {
            .stages {
                flex-direction: column;
                gap: 60px;
            }
            .pipeline-tube {
                display: none;
            }
            .stage {
                width: 100%;
                max-width: 400px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Donor Prediction Pipeline</h1>
            <p>End-to-End Deep Learning Workflow</p>
        </div>

        <div class="pipeline">
            <!-- Pipeline Tube (flowing background) -->
            <div class="pipeline-tube"></div>

            <!-- Pipeline Stages -->
            <div class="stages">
                <!-- Stage 1: Data Creation -->
                <div class="stage">
                    <div class="node">📊</div>
                    <div class="stage-label">Data Creation</div>
                    <div class="details">
                        <div class="detail-item">
                            <div class="detail-title">Generation</div>
                            <div class="detail-subtitle">500K Donors</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-title">SQL Export</div>
                            <div class="detail-subtitle">PostgreSQL</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-title">Parquet</div>
                            <div class="detail-subtitle">Optimized</div>
                        </div>
                    </div>
                </div>

                <!-- Stage 2: Feature Engineering -->
                <div class="stage">
                    <div class="node">🔧</div>
                    <div class="stage-label">Feature Engineering</div>
                    <div class="details">
                        <div class="detail-item">
                            <div class="detail-title">Network</div>
                            <div class="detail-subtitle">PageRank</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-title">Temporal</div>
                            <div class="detail-subtitle">RFM Analysis</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-title">Engagement</div>
                            <div class="detail-subtitle">Campaigns</div>
                        </div>
                    </div>
                </div>

                <!-- Stage 3: Model Training -->
                <div class="stage">
                    <div class="node">🤖</div>
                    <div class="stage-label">Model Training</div>
                    <div class="details">
                        <div class="detail-item">
                            <div class="detail-title">Data Split</div>
                            <div class="detail-subtitle">2021-2023</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-title">GNN Training</div>
                            <div class="detail-subtitle">Multimodal</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-title">Tuning</div>
                            <div class="detail-subtitle">Optimization</div>
                        </div>
                    </div>
                </div>

                <!-- Stage 4: Evaluation -->
                <div class="stage">
                    <div class="node">📈</div>
                    <div class="stage-label">Evaluation</div>
                    <div class="details">
                        <div class="detail-item">
                            <div class="detail-title">Validation</div>
                            <div class="detail-subtitle">Time-based</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-title">Metrics</div>
                            <div class="detail-subtitle">AUC: 94.88%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-title">Analysis</div>
                            <div class="detail-subtitle">SHAP Values</div>
                        </div>
                    </div>
                </div>

                <!-- Stage 5: Dashboard -->
                <div class="stage">
                    <div class="node">📱</div>
                    <div class="stage-label">Dashboard</div>
                    <div class="details">
                        <div class="detail-item">
                            <div class="detail-title">Streamlit</div>
                            <div class="detail-subtitle">5 Pages</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-title">Interactive</div>
                            <div class="detail-subtitle">Real-time</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-title">Insights</div>
                            <div class="detail-subtitle">Visual</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Stats Dashboard -->
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Dataset</div>
                <div class="stat-value">500K</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Features</div>
                <div class="stat-value">50+</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Model Type</div>
                <div class="stat-value" style="font-size: 1.4em;">Multimodal Fusion</div>
                <div style="font-size: 0.7em; margin-top: 5px; opacity: 0.9;">RNN + CNN + MLP + LSTM</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">AUC Score</div>
                <div class="stat-value">94.88%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Pages</div>
                <div class="stat-value">5</div>
            </div>
        </div>

        <div class="info">
            💡 Animated pipeline flow • Hover nodes for effects • Click stats for interactions
        </div>
    </div>
</body>
</html>
"""


def generate_pipeline_html():
    """Generate the pipeline visualization HTML file"""
    output_file = 'project_pipeline_visualization.html'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(HTML_CONTENT)
    
    print(f"✅ Generated {output_file}")
    print(f"📂 Location: {output_file}")
    print(f"\n🌐 Open in browser: file://{output_file}")
    print("✨ Interactive animated pipeline visualization ready!")


if __name__ == "__main__":
    generate_pipeline_html()
