/**
 * LoadingSpinner - 加载状态组件
 */
import React from 'react';

export function LoadingSpinner({ 
  message = 'Loading...', 
  size = 'medium',
  fullScreen = false 
}) {
  const sizes = {
    small: '16px',
    medium: '32px',
    large: '48px'
  };

  const spinnerStyle = {
    width: sizes[size],
    height: sizes[size],
    border: '3px solid rgba(255,255,255,0.1)',
    borderTopColor: '#00d2ff',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite'
  };

  const containerStyle = fullScreen ? {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(10, 10, 12, 0.9)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 9999
  } : {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '40px'
  };

  return (
    <div style={containerStyle}>
      <div style={spinnerStyle} />
      {message && (
        <div style={{ 
          marginTop: '16px', 
          color: '#888', 
          fontSize: '14px' 
        }}>
          {message}
        </div>
      )}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default LoadingSpinner;
