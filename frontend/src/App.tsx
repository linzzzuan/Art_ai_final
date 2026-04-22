import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import AppLayout from './components/Layout/AppLayout';
import ErrorBoundary from './components/ErrorBoundary';
import DetectionPage from './pages/Detection';
import DatasetPage from './pages/Dataset';
import ResultsPage from './pages/Results';

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<Navigate to="/detection" replace />} />
            <Route path="/detection" element={<DetectionPage />} />
            <Route path="/dataset" element={<DatasetPage />} />
            <Route path="/results" element={<ResultsPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
