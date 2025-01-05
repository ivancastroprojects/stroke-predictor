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
            opacity: 0
          }}
        />
      ))}
    </div>
  );
} 