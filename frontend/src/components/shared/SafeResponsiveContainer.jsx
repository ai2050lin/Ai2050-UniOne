import React, { useEffect, useRef, useState } from 'react';
import { ResponsiveContainer } from 'recharts';

export default function SafeResponsiveContainer({
  children,
  minHeight = 220,
  minWidth = 0,
  style,
  placeholder = null,
}) {
  const hostRef = useRef(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) {
      return undefined;
    }

    const updateReady = () => {
      const nextReady = host.clientWidth > 0 && host.clientHeight > 0;
      setReady((prev) => (prev === nextReady ? prev : nextReady));
    };

    updateReady();

    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(() => {
        updateReady();
      });
      observer.observe(host);
      return () => observer.disconnect();
    }

    window.addEventListener('resize', updateReady);
    return () => window.removeEventListener('resize', updateReady);
  }, []);

  return (
    <div
      ref={hostRef}
      style={{
        width: '100%',
        height: '100%',
        minWidth,
        minHeight,
        ...style,
      }}
    >
      {ready ? (
        <ResponsiveContainer width="100%" height="100%" minWidth={minWidth} minHeight={minHeight}>
          {children}
        </ResponsiveContainer>
      ) : (
        placeholder
      )}
    </div>
  );
}
