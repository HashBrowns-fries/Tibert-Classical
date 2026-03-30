import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Header } from './components/Header';
import { AnalyzerPage } from './pages/AnalyzerPage';
import { LearnerPage } from './pages/LearnerPage';
import { CorpusPage } from './pages/CorpusPage';
import { LookupPage } from './pages/LookupPage';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen" style={{ fontFamily: 'var(--font-sans)' }}>
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<AnalyzerPage />} />
            <Route path="/dict" element={<LookupPage />} />
            <Route path="/learner" element={<LearnerPage />} />
            <Route path="/corpus" element={<CorpusPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
