import React from 'react';
import { Box, List, ListItemButton, ListItemText, Typography, IconButton, Tooltip } from '@mui/material';
import { FiMenu, FiChevronRight } from 'react-icons/fi';

const Sidebar = ({ isOpen, onToggle, onMenuClick, activeMenuId = 'dashboard' }) => {
  const menuItems = [
    { id: 'dashboard', label: '대시보드' },
    { id: 'prediction', label: 'AI 예측' },
    { id: 'news', label: '감염병 뉴스' },
    { id: 'weekly', label: '주간 발생 동향' },
    { id: 'influenza', label: '인플루엔자란?' },
    { id: 'hospital', label: '근처 병원찾기' },
  ];

  return (
    <Box
      sx={{
        width: isOpen ? 240 : 64,
        height: '100vh',
        backgroundColor: '#1570ef',
        display: 'flex',
        flexDirection: 'column',
        position: 'fixed',
        left: 0,
        top: 0,
        zIndex: 1000,
        transition: 'width 0.3s ease',
        overflow: 'hidden',
      }}
    >
      {/* 로고 영역 */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: isOpen ? 'space-between' : 'center',
          p: 2,
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <IconButton
            onClick={onToggle}
            sx={{
              color: '#fff',
              p: 0.5,
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
              },
            }}
          >
            <FiMenu size={20} />
          </IconButton>
          {isOpen && (
            <Typography
              variant="h6"
              sx={{
                color: '#fff',
                fontWeight: 700,
                fontSize: '1.1rem',
                whiteSpace: 'nowrap',
              }}
            >
              LOGO
            </Typography>
          )}
        </Box>
        {isOpen && (
          <Box
            sx={{
              width: 32,
              height: 32,
              borderRadius: '50%',
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Box
              sx={{
                width: 20,
                height: 20,
                borderRadius: '50%',
                backgroundColor: '#3b82f6',
              }}
            />
          </Box>
        )}
      </Box>

      {/* 메뉴 항목 */}
      <List sx={{ flex: 1, p: 1, pt: 2 }}>
        {menuItems.map((item) => {
          const isActive = item.id === activeMenuId;
          return (
          <Tooltip key={item.id} title={!isOpen ? item.label : ''} placement="right">
            <ListItemButton
              onClick={() => onMenuClick && onMenuClick(item.id)}
              sx={{
                mb: 0.5,
                borderRadius: 1,
                backgroundColor: isActive ? '#3b82f6' : 'transparent',
                color: isActive ? '#fff' : 'rgba(255, 255, 255, 0.7)',
                '&:hover': {
                  backgroundColor: isActive ? '#3b82f6' : 'rgba(255, 255, 255, 0.1)',
                  color: '#fff',
                },
                py: 1.5,
                px: isOpen ? 2 : 1.5,
                justifyContent: isOpen ? 'flex-start' : 'center',
                minHeight: 48,
              }}
            >
              {isOpen ? (
                <>
                  <ListItemText
                    primary={item.label}
                    primaryTypographyProps={{
                      fontSize: '0.9rem',
                      fontWeight: isActive ? 600 : 400,
                    }}
                  />
                  {item.hasArrow && (
                    <FiChevronRight size={16} style={{ marginLeft: 'auto' }} />
                  )}
                </>
              ) : (
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {/* 아이콘만 표시할 수 있도록 공간 확보 */}
                </Box>
              )}
            </ListItemButton>
          </Tooltip>
          );
        })}
      </List>

      {/* 로그아웃 버튼 */}
      <Box sx={{ p: 2, borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
        <Tooltip title={!isOpen ? '로그아웃' : ''} placement="right">
          <ListItemButton
            sx={{
              borderRadius: 1,
              backgroundColor: 'transparent',
              color: 'rgba(255, 255, 255, 0.7)',
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                color: '#fff',
              },
              py: 1.5,
              px: isOpen ? 2 : 1.5,
              justifyContent: isOpen ? 'flex-start' : 'center',
              minHeight: 48,
            }}
          >
            {isOpen && (
              <ListItemText
                primary="로그아웃"
                primaryTypographyProps={{
                  fontSize: '0.9rem',
                  fontWeight: 400,
                }}
              />
            )}
          </ListItemButton>
        </Tooltip>
      </Box>
    </Box>
  );
};

export default Sidebar;
