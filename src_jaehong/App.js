import React, { useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import Prediction from './components/Prediction';
import Footer from './components/Footer';
import './App.css';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1e3a8a',
      light: '#3b82f6',
      dark: '#1e40af',
    },
    secondary: {
      main: '#0369a1',
    },
    background: {
      default: '#f3f4f6',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: [
      'Pretendard',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Malgun Gothic"',
      '맑은 고딕',
      'sans-serif',
    ].join(','),
    fontSize: 14,
  },
  components: {
    MuiTableHead: {
      styleOverrides: {
        root: {
          backgroundColor: '#f1f5f9',
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        head: {
          fontWeight: 600,
          backgroundColor: '#f1f5f9',
          borderBottom: '2px solid #cbd5e1',
        },
      },
    },
  },
});

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeMenuId, setActiveMenuId] = useState('dashboard');
  const [shouldOpenHospitalMap, setShouldOpenHospitalMap] = useState(false);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const handleMenuClick = (menuId) => {
    if (menuId === 'hospital') {
      setActiveMenuId('dashboard');
      setShouldOpenHospitalMap(true);
    } else if (menuId === 'news' || menuId === 'weekly' || menuId === 'influenza') {
      // Dashboard 컴포넌트에서 다이얼로그를 열도록 activeMenuId를 'dashboard'로 설정
      setActiveMenuId('dashboard');
    } else {
      setActiveMenuId(menuId);
    }
  };

  const handleHospitalMapOpened = () => {
    setShouldOpenHospitalMap(false);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div className="App">
        <Sidebar 
          isOpen={sidebarOpen} 
          onToggle={toggleSidebar}
          onMenuClick={handleMenuClick}
          activeMenuId={activeMenuId}
        />
        <Header isOpen={sidebarOpen} />
        {activeMenuId === 'prediction' ? (
          <Prediction isOpen={sidebarOpen} />
        ) : (
          <Dashboard 
            isOpen={sidebarOpen}
            shouldOpenHospitalMap={shouldOpenHospitalMap}
            onHospitalMapOpened={handleHospitalMapOpened}
            activeMenuId={activeMenuId}
          />
        )}
        <Footer isOpen={sidebarOpen} />
      </div>
    </ThemeProvider>
  );
}

export default App;

