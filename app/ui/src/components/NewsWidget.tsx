'use client';
import React, { useEffect, useRef } from 'react';

function NewsWidget() {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!container.current || container.current.hasChildNodes()) {
      return;
    }

    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-timeline.js';
    script.type = 'text/javascript';
    script.async = true;
    script.innerHTML = JSON.stringify({
      "feedMode": "all_symbols",
      "colorTheme": "dark",
      "isTransparent": false,
      "displayMode": "regular",
      "width": "100%",
      "height": "100%",
      "locale": "en"
    });
    
    container.current.appendChild(script);

  }, []);

  return (
    <div 
      ref={container} 
      style={{ height: "100%", width: "100%" }}
    />
  );
}

export default React.memo(NewsWidget); 