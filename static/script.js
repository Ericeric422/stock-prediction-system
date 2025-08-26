document.addEventListener('DOMContentLoaded', () => {
    // Form elements
    const predictForm = document.getElementById('predict-form');
    const symbolInput = document.getElementById('symbol-input');
    
    // Results elements
    const resultsContainer = document.getElementById('result-container');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorContainer = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');
    const explainButton = document.getElementById('explain-button');
    const explanationContainer = document.getElementById('explanation-container');
    const modelSelect = document.getElementById('model-select');
    const suggestionsEl = document.getElementById('symbol-suggestions');

    // Lightweight ticker list for autocomplete (symbol, name)
    const TICKERS = [
        { s: 'AAPL', n: 'Apple Inc.' },
        { s: 'MSFT', n: 'Microsoft Corporation' },
        { s: 'GOOGL', n: 'Alphabet Inc. (Class A)' },
        { s: 'GOOG', n: 'Alphabet Inc. (Class C)' },
        { s: 'AMZN', n: 'Amazon.com, Inc.' },
        { s: 'META', n: 'Meta Platforms, Inc.' },
        { s: 'TSLA', n: 'Tesla, Inc.' },
        { s: 'NVDA', n: 'NVIDIA Corporation' },
        { s: 'BRK.B', n: 'Berkshire Hathaway Inc. Class B' },
        { s: 'BRK.A', n: 'Berkshire Hathaway Inc. Class A' },
        { s: 'JPM', n: 'JPMorgan Chase & Co.' },
        { s: 'V', n: 'Visa Inc.' },
        { s: 'JNJ', n: 'Johnson & Johnson' },
        { s: 'WMT', n: 'Walmart Inc.' },
        { s: 'PG', n: 'Procter & Gamble Company' },
        { s: 'HD', n: 'Home Depot, Inc.' },
        { s: 'MA', n: 'Mastercard Incorporated' },
        { s: 'XOM', n: 'Exxon Mobil Corporation' },
        { s: 'BAC', n: 'Bank of America Corporation' },
        { s: 'PFE', n: 'Pfizer Inc.' },
        { s: 'KO', n: 'Coca-Cola Company' },
        { s: 'PEP', n: 'PepsiCo, Inc.' },
        { s: 'DIS', n: 'Walt Disney Company' },
        { s: 'NFLX', n: 'Netflix, Inc.' },
        { s: 'INTC', n: 'Intel Corporation' },
        { s: 'CSCO', n: 'Cisco Systems, Inc.' },
        { s: 'ADBE', n: 'Adobe Inc.' },
        { s: 'CRM', n: 'Salesforce, Inc.' },
        { s: 'ORCL', n: 'Oracle Corporation' },
        { s: 'T', n: 'AT&T Inc.' },
        { s: 'TM', n: 'Toyota Motor Corporation' },
        { s: 'NKE', n: 'NIKE, Inc.' },
        { s: 'MCD', n: 'McDonald\'s Corporation' },
        { s: 'ABNB', n: 'Airbnb, Inc.' },
        { s: 'UBER', n: 'Uber Technologies, Inc.' },
        { s: 'LYFT', n: 'Lyft, Inc.' },
        { s: 'SQ', n: 'Block, Inc. (Square)' },
        { s: 'SHOP', n: 'Shopify Inc.' },
        { s: 'BABA', n: 'Alibaba Group Holding Limited' },
        { s: 'AMD', n: 'Advanced Micro Devices, Inc.' },
        { s: 'QCOM', n: 'Qualcomm Incorporated' },
        { s: 'SPY', n: 'SPDR S&P 500 ETF Trust' },
        { s: 'QQQ', n: 'Invesco QQQ Trust' }
    ];

    let activeIndex = -1; // keyboard navigation in suggestions

    function hideSuggestions() {
        if (suggestionsEl) {
            suggestionsEl.classList.add('hidden');
            suggestionsEl.innerHTML = '';
            activeIndex = -1;
        }
    }

    function selectSuggestion(symbol) {
        symbolInput.value = symbol;
        hideSuggestions();
        symbolInput.focus();
    }

    function renderSuggestions(list) {
        if (!suggestionsEl) return;
        if (!list || list.length === 0) {
            hideSuggestions();
            return;
        }
        suggestionsEl.innerHTML = '';
        list.slice(0, 8).forEach((item, idx) => {
            const div = document.createElement('div');
            div.className = 'suggestion-item' + (idx === activeIndex ? ' active' : '');
            div.dataset.symbol = item.s;
            div.innerHTML = `<span class="suggestion-symbol">${item.s}</span><span class="suggestion-name">${item.n}</span>`;
            div.addEventListener('mousedown', (e) => {
                e.preventDefault(); // prevent input blur before click
                selectSuggestion(item.s);
            });
            suggestionsEl.appendChild(div);
        });
        suggestionsEl.classList.remove('hidden');
    }

    // Lightweight Levenshtein distance (small strings; OK for this dataset)
    function levenshtein(a, b) {
        const m = a.length, n = b.length;
        if (m === 0) return n;
        if (n === 0) return m;
        const dp = new Array(n + 1);
        for (let j = 0; j <= n; j++) dp[j] = j;
        for (let i = 1; i <= m; i++) {
            let prev = i - 1;
            dp[0] = i;
            for (let j = 1; j <= n; j++) {
                const tmp = dp[j];
                const cost = a[i - 1] === b[j - 1] ? 0 : 1;
                dp[j] = Math.min(
                    dp[j] + 1,        // deletion
                    dp[j - 1] + 1,    // insertion
                    prev + cost       // substitution
                );
                prev = tmp;
            }
        }
        return dp[n];
    }

    function getFilteredSuggestions(query) {
        const q = query.trim().toUpperCase();
        if (!q) return [];

        const scored = [];
        for (const t of TICKERS) {
            const sym = t.s.toUpperCase();
            const name = t.n.toUpperCase();
            let score = Infinity;

            if (sym === q) {
                score = -1; // best possible
            }
            if (sym.startsWith(q)) {
                // tighter starts-with and shorter remainder preferred
                score = Math.min(score, 0 + (sym.length - q.length) * 0.01);
            }
            if (sym.includes(q)) {
                score = Math.min(score, 0.5 + sym.indexOf(q) * 0.02);
            }
            if (name.includes(q)) {
                score = Math.min(score, 1 + name.indexOf(q) * 0.02);
            }
            // small fuzzy match tolerance on symbol only
            const dist = levenshtein(sym, q);
            if (dist <= 2) {
                score = Math.min(score, 2 + dist * 0.3);
            }

            if (score < Infinity) {
                scored.push({ t, score });
            }
        }

        scored.sort((a, b) => {
            if (a.score !== b.score) return a.score - b.score;
            // tie-breakers: shorter symbol, then alpha
            if (a.t.s.length !== b.t.s.length) return a.t.s.length - b.t.s.length;
            return a.t.s.localeCompare(b.t.s);
        });

        return scored.map(s => s.t);
    }

    function handleInputChange() {
        const val = symbolInput.value;
        const list = getFilteredSuggestions(val);
        activeIndex = -1;
        renderSuggestions(list);
    }

    function handleInputKeyDown(e) {
        if (suggestionsEl && !suggestionsEl.classList.contains('hidden')) {
            const items = Array.from(suggestionsEl.querySelectorAll('.suggestion-item'));
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                activeIndex = (activeIndex + 1) % items.length;
                renderSuggestions(items.map(it => ({ s: it.dataset.symbol, n: it.querySelector('.suggestion-name').textContent })));
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                activeIndex = (activeIndex - 1 + items.length) % items.length;
                renderSuggestions(items.map(it => ({ s: it.dataset.symbol, n: it.querySelector('.suggestion-name').textContent })));
            } else if (e.key === 'Enter') {
                if (activeIndex >= 0 && activeIndex < items.length) {
                    e.preventDefault();
                    const sym = items[activeIndex].dataset.symbol;
                    selectSuggestion(sym);
                }
            } else if (e.key === 'Escape') {
                hideSuggestions();
            }
        }
    }

    // Initialize the app
    function init() {
        setupEventListeners();
        // Restore last selected model
        if (modelSelect) {
            const saved = localStorage.getItem('modelSelect');
            if (saved) modelSelect.value = saved;
        }
    }

    // Set up event listeners
    function setupEventListeners() {
        if (predictForm) {
            predictForm.addEventListener('submit', handleFormSubmit);
        }
        if (symbolInput) {
            symbolInput.addEventListener('input', handleInputChange);
            symbolInput.addEventListener('keydown', handleInputKeyDown);
            symbolInput.addEventListener('blur', () => setTimeout(hideSuggestions, 150));
        }
        // Click outside to close suggestions
        document.addEventListener('click', (e) => {
            if (!e.target.closest('#symbol-suggestions') && !e.target.closest('#symbol-input')) {
                hideSuggestions();
            }
        });
        if (explainButton) {
            explainButton.addEventListener('click', handleExplainClick);
        }
        if (modelSelect) {
            modelSelect.addEventListener('change', () => {
                localStorage.setItem('modelSelect', modelSelect.value);
            });
        }
    }

    // Initialize the app when DOM is loaded
    init();

    // Handle form submission
    function handleFormSubmit(e) {
        e.preventDefault();
        
        const symbol = symbolInput.value.trim().toUpperCase();
        if (!symbol) return;
        
        // Show loading state
        loadingSpinner.classList.remove('hidden');
        if (resultsContainer) resultsContainer.classList.add('hidden');
        if (errorContainer) errorContainer.classList.add('hidden');
        if (explanationContainer) explanationContainer.classList.add('hidden');
        hideSuggestions();
        
        // Make prediction
        const modelParam = modelSelect ? `?model=${encodeURIComponent(modelSelect.value)}` : '';
        fetch(`/predict/${symbol}${modelParam}`)
            .then(async response => {
                let data;
                try { data = await response.json(); } catch (e) { data = {}; }
                if (!response.ok) {
                    const msg = (data && data.error) ? data.error : `Request failed (${response.status})`;
                    showError(msg);
                    throw new Error(msg);
                }
                return data;
            })
            .then(data => {
                if (data.error) {
                    showError(data.error);
                } else {
                    displayPrediction(symbol, data);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                if (errorContainer) {
                    errorText.textContent = `Error: ${error.message || 'Failed to get prediction'}`;
                    errorContainer.classList.remove('hidden');
                }
            })
            .finally(() => {
                loadingSpinner.classList.add('hidden');
            });
    }

    // Handle explain button click
    function handleExplainClick() {
        const symbol = symbolInput.value.trim().toUpperCase();
        if (!symbol) return;
        
        if (explanationContainer) {
            explanationContainer.innerHTML = '<div class="loading-text">Generating explanation...</div>';
            explanationContainer.classList.remove('hidden');
            
            const modelParam = modelSelect ? `?model=${encodeURIComponent(modelSelect.value)}` : '';
            fetch(`/explain/${symbol}${modelParam}`)
                .then(async response => {
                    let data;
                    try { data = await response.json(); } catch (e) { data = {}; }
                    if (!response.ok) {
                        const msg = (data && data.error) ? data.error : `Request failed (${response.status})`;
                        throw new Error(msg);
                    }
                    return data;
                })
                .then(data => {
                    if (data.error) {
                        explanationContainer.innerHTML = `<p class="error">Error: ${data.error}</p>`;
                    } else {
                        explanationContainer.innerHTML = data.explanation || 'No explanation available';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    explanationContainer.innerHTML = `Error: ${error.message || 'Failed to generate explanation'}`;
                });
        }
    }

    // Display prediction results
    function displayPrediction(symbol, data) {
        resultsContainer.classList.remove('hidden');
        errorContainer.classList.add('hidden');

        const symbolEl = document.querySelector('.symbol');
        symbolEl.textContent = data.company_name
            ? `${data.symbol.toUpperCase()} — ${data.company_name}`
            : data.symbol.toUpperCase();
        document.getElementById('model-name').textContent = data.model_used || 'N/A';
        document.getElementById('confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
        
        const predictionDate = new Date(data.timestamp);
        document.getElementById('prediction-date').textContent = predictionDate.toLocaleDateString();
        document.getElementById('last-updated').textContent = predictionDate.toLocaleTimeString();

        const directionContainer = document.querySelector('.prediction-direction');
        const directionIcon = directionContainer.querySelector('i');
        const directionText = directionContainer.querySelector('span');

        if (data.direction === 'UP') {
            directionContainer.className = 'prediction-direction direction-up';
            directionIcon.className = 'fas fa-arrow-up';
            directionText.textContent = 'UP';
        } else {
            directionContainer.className = 'prediction-direction direction-down';
            directionIcon.className = 'fas fa-arrow-down';
            directionText.textContent = 'DOWN';
        }

        // Render chart if chart data is available
        try {
            if (data.chart && Array.isArray(data.chart.labels)) {
                const ctx = document.getElementById('price-chart').getContext('2d');
                // Dispose existing chart if re-rendering
                if (window.__priceChart) {
                    window.__priceChart.destroy();
                }
                window.__priceChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.chart.labels,
                        datasets: [
                            {
                                label: 'Close',
                                data: data.chart.close || [],
                                borderColor: 'rgba(52, 152, 219, 1)',
                                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                                borderWidth: 2,
                                pointRadius: 0,
                                tension: 0.2,
                                spanGaps: true
                            },
                            {
                                label: 'SMA 20',
                                data: data.chart.sma20 || [],
                                borderColor: 'rgba(46, 204, 113, 1)',
                                backgroundColor: 'rgba(46, 204, 113, 0.1)',
                                borderWidth: 1.5,
                                pointRadius: 0,
                                tension: 0.2,
                                spanGaps: true
                            },
                            {
                                label: 'SMA 50',
                                data: data.chart.sma50 || [],
                                borderColor: 'rgba(155, 89, 182, 1)',
                                backgroundColor: 'rgba(155, 89, 182, 0.1)',
                                borderWidth: 1.5,
                                pointRadius: 0,
                                tension: 0.2,
                                spanGaps: true
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: { mode: 'index', intersect: false },
                        plugins: {
                            legend: { display: true },
                            tooltip: { enabled: true }
                        },
                        scales: {
                            x: {
                                ticks: {
                                    maxTicksLimit: 8,
                                    autoSkip: true
                                }
                            },
                            y: {
                                beginAtZero: false,
                                ticks: {
                                    callback: (v) => typeof v === 'number' ? v.toFixed(2) : v
                                }
                            }
                        }
                    }
                });
            }
        } catch (e) {
            console.warn('Chart render skipped:', e);
        }
    }

    // Show error message
    function showError(message) {
        errorText.textContent = `Error: ${message}. Please check the symbol and try again.`;
        errorContainer.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
    }

    // Hide error message
    function hideError() {
        errorContainer.classList.add('hidden');
    }
});
