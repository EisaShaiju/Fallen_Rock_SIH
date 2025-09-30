import React from 'react';

interface HtmlFileViewerProps {
  fileUrl: string; // The URL to the HTML file in the public folder
}

const HtmlFileViewer: React.FC<HtmlFileViewerProps> = ({ fileUrl }) => {
  return (
    <iframe
      src={fileUrl}
      title="HTML File Viewer"
      style={{
        width: '100%',
        height: '100%',
        border: 'none', // Optional: removes the default iframe border
      }}
    />
  );
};

export default HtmlFileViewer;