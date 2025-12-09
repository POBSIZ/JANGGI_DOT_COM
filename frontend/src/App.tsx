import { BrowserRouter, Routes, Route } from 'react-router-dom';
import GamePage from './pages/GamePage';
import MultiplayerPage from './pages/MultiplayerPage';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<GamePage />} />
        <Route path="/multiplayer" element={<MultiplayerPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
