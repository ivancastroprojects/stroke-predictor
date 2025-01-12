import React from 'react';

export default function BackgroundRays() {
  return (
    <div className="background-container">
      {[...Array(18)].map((_, index) => (
        <div
          key={index}
          className={`run ${index % 2 === 0 ? 'left' : 'right'}`}
          style={{
            animationDelay: `${index * 0.3}s`,
            opacity: 0,
            background: index % 2 === 0 
              ? 'linear-gradient(90deg, transparent, rgba(0, 183, 255, 0.4))'
              : 'linear-gradient(270deg, transparent, rgba(0, 255, 213, 0.4))'
          }}
        />
      ))}
    </div>
  );
} 