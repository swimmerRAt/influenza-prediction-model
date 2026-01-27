import React from 'react';
import { Box, Typography, IconButton } from '@mui/material';
import { FiUser, FiChevronDown } from 'react-icons/fi';

const Header = ({ isOpen }) => {
  return (
    <Box
      sx={{
        height: 60,
        backgroundColor: '#ffffff',
        borderBottom: '1px solid rgba(0, 0, 0, 0.1)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        px: 3,
        position: 'fixed',
        top: 0,
        left: isOpen ? 240 : 64,
        right: 0,
        zIndex: 999,
        transition: 'left 0.3s ease',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          cursor: 'pointer',
        }}
      >
        <Box
          sx={{
            width: 32,
            height: 32,
            borderRadius: '50%',
            backgroundColor: '#e5e7eb',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <FiUser size={18} color="#6b7280" />
        </Box>
        <Typography
          variant="body1"
          sx={{
            color: '#1f2937',
            fontWeight: 500,
            fontSize: '0.9rem',
          }}
        >
          Admin
        </Typography>
        <IconButton
          sx={{
            p: 0,
            color: '#6b7280',
          }}
        >
          <FiChevronDown size={16} />
        </IconButton>
      </Box>
    </Box>
  );
};

export default Header;
