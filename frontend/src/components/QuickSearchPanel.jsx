import React, { useState } from 'react';

const QuickSearchPanel = ({ onSearch }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [sortBy, setSortBy] = useState('time');

  const handleSearch = (e) => {
    setSearchTerm(e.target.value);
    onSearch({
      term: e.target.value,
      type: filterType,
      sortBy
    });
  };

  return (
    <div className="panel-section panel-compact">
      <h3 className="panel-title">搜索</h3>
      <div className="search-controls">
        <div className="search-input-group">
          <input
            type="text"
            className="search-input"
            placeholder="搜索..."
            value={searchTerm}
            onChange={handleSearch}
          />
        </div>
      </div>
    </div>
  );
};

export default QuickSearchPanel;
