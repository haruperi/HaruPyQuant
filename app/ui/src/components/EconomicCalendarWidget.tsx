'use client';

import React, { useEffect, useRef } from 'react';

function EconomicCalendarWidget() {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!container.current || container.current.hasChildNodes()) {
      return;
    }

    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-events.js';
    script.type = 'text/javascript';
    script.async = true;
    script.innerHTML = JSON.stringify({
      "colorTheme": "dark",
      "isTransparent": false,
      "width": "100%",
      "height": "100%",
      "locale": "en",
      "importance_filter": "-1,0,1"
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

export default React.memo(EconomicCalendarWidget); 