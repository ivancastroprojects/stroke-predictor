import React, { Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import MainComponent from './components/MainComponent';
import PredictionResults from './components/PredictionResults';
import BackgroundRays from './components/BackgroundRays';
import './App.css';

// Componente de carga para Suspense
const LoadingFallback = () => (
  <div className="loading-container">
    <div className="loading-spinner"></div>
    <p>Loading...</p>
  </div>
);

// Componente para manejar rutas no encontradas
const NotFound = () => {
  return (
    <div className="not-found-container">
      <h1>404 - Page Not Found</h1>
      <p>The page you are looking for does not exist.</p>
      <button onClick={() => window.location.href = '/'}>
        Return to Home
      </button>
    </div>
  );
};

// Componente de error boundary
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Routing Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-container">
          <h1>Something went wrong.</h1>
          <button onClick={() => window.location.href = '/'}>
            Return to Home
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

function App() {
  return (
    <Router>
      <ErrorBoundary>
        <div className="App">
          <BackgroundRays />
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              {/* Ruta raíz */}
              <Route path="/" element={<MainComponent />} />
              
              {/* Ruta de resultados */}
              <Route path="/results" element={<PredictionResults />} />
              
              {/* Redirección de rutas antiguas o alternativas */}
              <Route path="/prediction" element={<Navigate to="/results" replace />} />
              <Route path="/home" element={<Navigate to="/" replace />} />
              
              {/* Captura cualquier otra ruta */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </div>
      </ErrorBoundary>
    </Router>
  );
}

export default App;
