import React, { useId } from 'react';
import './ElectricCard.css';

export function ElectricCard({ children, className = '', color = '#00f0ff' }) {
  const filterId = useId().replace(/:/g, '');

  return (
    <div className={`electric-card-wrapper ${className}`}>
      <div 
        className="ec-glow-layer" 
        style={{ 
          borderColor: color, 
          filter: `url(#electric-${filterId}) blur(2px) drop-shadow(0 0 8px ${color})` 
        }}
      ></div>
      <div 
        className="ec-glow-layer inner" 
        style={{ 
          borderColor: color, 
          filter: `url(#electric-${filterId})` 
        }}
      ></div>
      <div className="ec-content-layer">
        {children}
      </div>
      
      <svg width="0" height="0" className="ec-svg-defs">
        <filter id={`electric-${filterId}`}>
          <feTurbulence type="fractalNoise" baseFrequency="0.03" numOctaves="2" result="noise">
            <animate attributeName="seed" values="1;5;10;15;20;25;30;35;40;45;50" dur="1s" repeatCount="indefinite" />
          </feTurbulence>
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="12" xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </svg>
    </div>
  );
}
