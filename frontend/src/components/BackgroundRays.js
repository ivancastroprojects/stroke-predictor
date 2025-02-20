import React from 'react';
import './BackgroundRays.css';

export default function BackgroundRays() {
  return (
    <div className="background-container">
      {[...Array(34)].map((_, index) => (
        <div
          key={index}
          className={`run ${index % 2 === 0 ? 'left' : 'right'}`}
          style={{
            animationDelay: `${index * 0.3}s`,
            background: index % 2 === 0 
              ? 'linear-gradient(90deg, transparent, rgba(0, 183, 255, 0.2))'
              : 'linear-gradient(90deg, rgba(0, 255, 213, 0.2), transparent)'
          }}
        />
      ))}
    </div>
  );
} 