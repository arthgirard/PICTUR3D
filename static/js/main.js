document.addEventListener("DOMContentLoaded", function() {
  const navbar = document.getElementById("navbar");
  
  function checkScroll() {
    if (window.scrollY > 10) {
      navbar.classList.add("scrolled");
    } else {
      navbar.classList.remove("scrolled");
    }
  }

  // Initial check
  checkScroll();
  
  // Add scroll event listener
  window.addEventListener("scroll", checkScroll);

  let simulationFinishedUI = false;
  let currentIteration = 0;
  let totalSimulations;

  const themeToggle = document.getElementById("themeToggle");
  const body = document.body;
  const favicon = document.getElementById("favicon");

  function invertFavicon() {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = favicon.href;
    img.onload = function() {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        data[i] = 255 - data[i];
        data[i + 1] = 255 - data[i + 1];
        data[i + 2] = 255 - data[i + 2];
      }
      ctx.putImageData(imageData, 0, 0);
      favicon.href = canvas.toDataURL("image/png");
    };
  }

  function updateThemeIcon() {
    if (body.classList.contains("dark-mode")) {
      themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
      favicon.href = "/static/img/favicon.png";
    } else {
      themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
      invertFavicon();
    }
  }

  function initializeTheme() {
    if (localStorage.getItem("theme") === "dark") {
      body.classList.add("dark-mode");
      navbar.classList.add("dark-mode");
    } else {
      body.classList.add("light-mode");
    }
    updateThemeIcon();
  }

  themeToggle.addEventListener("click", function() {
    if (body.classList.contains("dark-mode")) {
      body.classList.remove("dark-mode");
      body.classList.add("light-mode");
      navbar.classList.remove("dark-mode");
      localStorage.setItem("theme", "light");
    } else {
      body.classList.remove("light-mode");
      body.classList.add("dark-mode");
      navbar.classList.add("dark-mode");
      localStorage.setItem("theme", "dark");
    }
    updateThemeIcon();
  });

  const startButton = document.getElementById("startSimulation");
  const modeSelect = document.getElementById("modeSelect");
  const simulationStatus = document.getElementById("simulationStatus");
  const startDateInput = document.getElementById("startDate");
  const endDateInput = document.getElementById("endDate");
  const resetAgentBtn = document.getElementById("resetAgentBtn");

  const $parametersCard = $("#parametersCard");
  const $resultsCard = $("#resultsCard");
  const $logsCard = $("#logsCard");

  $resultsCard.hide();
  $logsCard.hide();

  let simulationRunning = localStorage.getItem("simulationRunning") === "true";
  let resultsInterval = null;
  let logsInterval = null;
  let firstLogReceived = false;
  let lastKnownLogsCount = 0;

  function waitForSimulationStop() {
    const stopInterval = setInterval(() => {
      fetch("/results")
        .then(response => response.json())
        .then(data => {
          if (data.finished) {
            clearInterval(stopInterval);
            startButton.disabled = false;
          }
        })
        .catch(err => console.error("Error waiting for simulation stop:", err));
    }, 1000);
  }

  function clearPreviousData() {
    $("#liveActions").empty();
    firstLogReceived = false;
    lastKnownLogsCount = 0;
  }

  function stopSimulationUI() {
    simulationRunning = false;
    localStorage.setItem("simulationRunning", "false");
    startButton.disabled = true;
  
    if (resultsInterval) { clearInterval(resultsInterval); resultsInterval = null; }
    if (logsInterval) { clearInterval(logsInterval); logsInterval = null; }
    $("#liveActions").empty();
    if (window.solPriceChart) { window.solPriceChart.destroy(); window.solPriceChart = null; }
    if (window.equityChart) { window.equityChart.destroy(); window.equityChart = null; }
    if (window.lossChart) { window.lossChart.destroy(); window.lossChart = null; }
    
    fetch("/stop_simulation", { method: "POST" });
    
    $parametersCard.slideDown(300);
    $resultsCard.slideUp(300);
    $logsCard.slideUp(300).promise().done(function() {
      startButton.textContent = "START";
      startButton.classList.remove("btn-danger");
      startButton.classList.add("btn-primary");
      simulationStatus.textContent = "";
      resetAgentBtn.style.display = "none";
      document.querySelector('.trading-mode-card').classList.remove('trading-running');
      waitForSimulationStop();
      startButton.disabled = false;
    });
  }

  function fetchResults() {
    fetch("/results")
      .then(response => response.json())
      .then(data => {
        updateCharts(data);
        let finalBalance = parseFloat(data.final_balance) || 0;
        let iteration = data.iteration || 1;
        const totalSimulationsValue = data.total_simulations || 1;
        
        if (iteration > currentIteration) {
          currentIteration = iteration;
          fetchAndUpdatePerformanceHistory();
        }
        
        if (firstLogReceived) {
          simulationStatus.innerHTML = `<strong>SIMULATION ${iteration}/${totalSimulationsValue}</strong> | ` +
            `<strong>BALANCE:</strong> $${finalBalance.toFixed(2)} | ` +
            `<strong>TRADES:</strong> ${data.num_trades} | ` +
            `<strong>RETURN:</strong> ${parseFloat(data.percentage_return) ? parseFloat(data.percentage_return).toFixed(2) : "0"}%`;
        }
        
        if (data.finished && !simulationFinishedUI) {
          simulationFinishedUI = true;
          stopSimulationUI();
          console.info("Simulation finalized automatically via final iteration.");
        }
      })
      .catch(err => {
        console.error("Error fetching results:", err);
        simulationStatus.textContent = "Error updating simulation status.";
      });
  }    
  
  function pollResults() {
    fetchResults();
    resultsInterval = setInterval(fetchResults, 1000);
  }

  function pollLiveLogs() {
    logsInterval = setInterval(() => {
      fetch("/live_logs")
        .then(response => response.json())
        .then(data => {
          if (!firstLogReceived && data && data.length > 0) {
            firstLogReceived = true;
            $resultsCard.slideDown(300);
            $logsCard.slideDown(300);
            startButton.style.display = "inline-block";
            resetAgentBtn.style.display = "inline-block";
            fetchResults();
          }
          updateLiveActions(data);
          lastKnownLogsCount = data.length;
        })
        .catch(err => console.error("Error fetching live logs:", err));
    }, 1000);
  }

  function updateCharts(data) {
    if (!data || !data.dates || data.dates.length === 0) {
      console.warn("No chart data available to update.");
      return;
    }
    
    // Debug log to verify buy_hold_equity data is present.
    console.log("Buy & Hold Equity Data:", data.buy_hold_equity);
    
    Chart.defaults.font.family = 'Geist Mono, monospace';
    const decimation = {
      enabled: true,
      algorithm: 'lttb',
      samples: 100
    };
  
    // Equity Chart: Update or create with two datasets.
    if (window.equityChart && window.equityChart.data) {
      window.equityChart.data.labels = data.dates;
      // Update the simulation equity curve.
      window.equityChart.data.datasets[0].data = data.asset_values;
      // If the second dataset doesn't exist, push it.
      if (window.equityChart.data.datasets.length < 2) {
        window.equityChart.data.datasets.push({
          label: "Buy & Hold Equity",
          data: data.buy_hold_equity,
          borderColor: "rgba(255, 99, 132, 1)",
          fill: false,
          pointRadius: 0,
          borderDash: [5, 5]
        });
      } else {
        window.equityChart.data.datasets[1].data = data.buy_hold_equity;
      }
      window.equityChart.update();
    } else {
      const ctx1 = document.getElementById("equityChart").getContext("2d");
      window.equityChart = new Chart(ctx1, {
        type: "line",
        data: {
          labels: data.dates,
          datasets: [
            {
              label: "EQUITY (USD)",
              data: data.asset_values,
              borderColor: "rgba(75, 192, 192, 1)",
              fill: false,
              pointRadius: 0
            },
            {
              label: "BUY & HOLD (USD)",
              data: data.buy_hold_equity,
              borderColor: "rgba(255, 99, 132, 1)",
              fill: false,
              pointRadius: 0,
              borderDash: [5, 5]
            }
          ]
        },
        options: {
          responsive: true
        },
        plugins: {
          decimation: decimation
        }
      });
    }
    
    // SOL Price Chart update remains unchanged.
    if (window.solPriceChart && window.solPriceChart.data) {
      window.solPriceChart.data.labels = data.dates;
      window.solPriceChart.data.datasets[0].data = data.sol_prices;
      if (data.trade_dates && data.trade_prices && data.trade_signals) {
        const tradeData = data.trade_dates.map((date, i) => ({
          x: date,
          y: data.trade_prices[i],
          signal: data.trade_signals[i]
        }));
        if (window.solPriceChart.data.datasets.length < 2) {
          window.solPriceChart.data.datasets.push({
            label: "TRADES",
            data: tradeData,
            type: "line",
            parsing: { xAxisKey: "x", yAxisKey: "y" },
            pointRadius: 5,
            pointBackgroundColor: tradeData.map(pt =>
              pt.signal.toLowerCase().startsWith("buy") ? "green" : "red"
            ),
            showLine: false
          });
        } else {
          window.solPriceChart.data.datasets[1].data = tradeData;
          window.solPriceChart.data.datasets[1].pointBackgroundColor = tradeData.map(pt =>
            pt.signal.toLowerCase().startsWith("buy") ? "green" : "red"
          );
        }
      }
      window.solPriceChart.update();
    } else {
      const ctx2 = document.getElementById("solPriceChart").getContext("2d");
      let tradeData = [];
      if (data.trade_dates && data.trade_prices && data.trade_signals) {
        tradeData = data.trade_dates.map((date, i) => ({
          x: date,
          y: data.trade_prices[i],
          signal: data.trade_signals[i]
        }));
      }
      window.solPriceChart = new Chart(ctx2, {
        type: "line",
        data: {
          labels: data.dates,
          datasets: [
            {
              label: "SOL PRICE (USD)",
              data: data.sol_prices,
              borderColor: "rgba(153, 102, 255, 1)",
              fill: false,
              pointRadius: 0
            },
            {
              label: "TRADES",
              data: tradeData,
              type: "scatter",
              parsing: { xAxisKey: "x", yAxisKey: "y" },
              pointRadius: 5,
              pointBackgroundColor: tradeData.map(pt =>
                pt.signal.toLowerCase().startsWith("buy") ? "green" : "red"
              ),
              showLine: false
            }
          ]
        },
        options: {
          responsive: true,
          scales: {
            x: {
              type: "time",
              time: {
                parser: "YYYY-MM-DD HH:mm",
                tooltipFormat: "YYYY-MM-DD HH:mm"
              }
            }
          }
        }
      });
    }
    
    // Loss Chart update remains unchanged.
    if (window.lossChart && window.lossChart.data) {
      window.lossChart.data.labels = data.losses.map((_, i) => i + 1);
      window.lossChart.data.datasets[0].data = data.losses;
      window.lossChart.update();
    } else {
      const ctx3 = document.getElementById("lossChart").getContext("2d");
      window.lossChart = new Chart(ctx3, {
        type: "line",
        data: {
          labels: data.losses.map((_, i) => i + 1),
          datasets: [{
            label: "TRAINING LOSS",
            data: data.losses,
            borderColor: "rgba(255, 99, 132, 1)",
            fill: false
          }]
        },
        options: {responsive: true, animation: false},
        plugins: {
          decimation: decimation
        }
      });
    }
  }  
  
  function fetchAndUpdatePerformanceHistory() {
    fetch("/agent_performance")
      .then(response => response.json())
      .then(history => {
        const labels = history.map(item => item.timestamp);
        const netProfits = history.map(item => item.net_profit);

        // compute least‑squares trend line
        const n = netProfits.length;
        const xs = netProfits.map((_, i) => i);
        let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
        for (let i = 0; i < n; i++) {
          const x = xs[i], y = netProfits[i];
          sumX += x;
          sumY += y;
          sumXY += x * y;
          sumXX += x * x;
        }
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        const trendData = xs.map(x => intercept + slope * x);

        if (window.performanceHistoryChart) {
          const ds = window.performanceHistoryChart.data.datasets;
          ds[0].data = netProfits;
          if (ds.length < 2) {
            ds.push({
              label: "TREND LINE",
              data: trendData,
              borderColor: "rgba(255, 206, 86, 1)",
              borderDash: [5, 5],
              fill: false,
              pointRadius: 0
            });
          } else {
            ds[1].data = trendData;
          }
          window.performanceHistoryChart.data.labels = labels;
          window.performanceHistoryChart.update();
        } else {
          const ctx = document.getElementById("agentPerformanceChart").getContext("2d");
          window.performanceHistoryChart = new Chart(ctx, {
            type: "line",
            data: {
              labels: labels,
              datasets: [
                {
                  label: "NET PROFIT OVER TIME",
                  data: netProfits,
                  borderColor: "rgba(75, 192, 192, 1)",
                  fill: false,
                  pointRadius: 0
                },
                {
                  label: "TREND LINE",
                  data: trendData,
                  borderColor: "rgba(255, 206, 86, 1)",
                  borderDash: [5, 5],
                  fill: false,
                  pointRadius: 0
                }
              ]
            },
            options: {
              responsive: true,
              title: {
                display: true,
                text: "HISTORICAL AGENT PERFORMANCE"
              }
            }
          });
        }
      })
      .catch(err => console.error("Error fetching performance history:", err));
  }

  function updateLiveActions(logs) {
    if (!simulationRunning) return;
    const liveActionsDiv = $("#liveActions");
    liveActionsDiv.empty();
    logs.slice(-20).forEach(log => {
      $("<div>").text(log).appendTo(liveActionsDiv);
    });
    liveActionsDiv.scrollTop(liveActionsDiv.prop("scrollHeight"));
  }

  startButton.addEventListener("click", function() {
    if (!simulationRunning) {
      simulationFinishedUI = false;
      totalSimulations = parseInt(document.getElementById("numSimulations").value) || 1;
      
      $("#liveActions").empty();
      clearPreviousData();
      fetch("/clear_logs", { method: "POST" })
        .then(() => {
          simulationRunning = true;
          localStorage.setItem("simulationRunning", "true");
          startButton.textContent = "STOP";
          startButton.classList.remove("btn-primary");
          startButton.classList.add("btn-danger");
          startButton.style.display = "none";
          document.querySelector('.trading-mode-card').classList.add('trading-running');
          $parametersCard.slideUp(300);
          $resultsCard.hide();
          $logsCard.hide();
          simulationStatus.innerHTML = "INITIALIZING AGENT <i class='fas fa-spinner fa-spin'></i>";
          const mode = modeSelect.value;
          const startDate = startDateInput.value;
          const endDate = endDateInput.value;
          const numSimulations = document.getElementById("numSimulations").value;
          const saveResults = document.getElementById("saveResults").checked;
    
          // pick correct back‑end depending on the dropdown
          const endpoint = mode === "paper" ? "/start_paper" : "/start_simulation";
    
          fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              mode: mode,
              start_date: startDate,
              end_date: endDate,
              number_of_simulations: numSimulations,
              save_graphs: saveResults
            })
          })    
          .then(response => response.json())
          .then(data => {
            pollResults();
            pollLiveLogs();
            fetchAndUpdatePerformanceHistory();
          });
        });
    } else {
      stopSimulationUI();
    }
  });

  let resetAgentBtnDisabled = false;
  resetAgentBtn.addEventListener("click", function() {
    if (resetAgentBtnDisabled) return;
    resetAgentBtnDisabled = true;
    resetAgentBtn.disabled = true;
    startButton.style.display = "none";
    if (confirm("Are you sure you want to reset the agent? This will stop any running simulation and delete the agent, scaler, replay buffer, and performance history.")) {
      fetch("/delete_agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirmation: true })
      })
      .then(response => response.json())
      .then(data => {
        alert(data.status);
        if (resultsInterval) { clearInterval(resultsInterval); resultsInterval = null; }
        if (logsInterval) { clearInterval(logsInterval); logsInterval = null; }
        simulationRunning = false;
        localStorage.setItem("simulationRunning", "false");
        $parametersCard.slideDown(300);
        $resultsCard.slideUp(300);
        $logsCard.slideUp(300);
        startButton.textContent = "START";
        startButton.classList.remove("btn-danger");
        startButton.classList.add("btn-primary");
        simulationStatus.textContent = "";
        $("#liveActions").empty();
        resetAgentBtn.style.display = "none";
      })
      .catch(err => {
        console.error("Error resetting agent:", err);
        alert("An error occurred while resetting the agent.");
      })
      .finally(() => {
        resetAgentBtnDisabled = false;
        resetAgentBtn.disabled = false;
        startButton.style.display = "inline-block";
      });
    } else {
      resetAgentBtnDisabled = false;
      resetAgentBtn.disabled = false;
      startButton.style.display = "inline-block";
    }
  });

  if (simulationRunning) {
    startButton.textContent = "STOP";
    startButton.classList.remove("btn-primary");
    startButton.classList.add("btn-danger");
    document.querySelector('.trading-mode-card').classList.add('trading-running');
    $parametersCard.hide();
    resetAgentBtn.style.display = "inline-block";
    fetch("/live_logs")
      .then(response => response.json())
      .then(data => {
        if (data && data.length > 0) {
          $resultsCard.show();
          $logsCard.show();
          firstLogReceived = true;
          updateLiveActions(data);
          startButton.style.display = "inline-block";
        }
      })
      .catch(err => console.error("Error fetching initial logs:", err))
      .finally(() => {
        fetchAndUpdatePerformanceHistory();
        pollResults();
        pollLiveLogs();
      });
  } else {
    startButton.textContent = "START";
    startButton.classList.remove("btn-danger");
    startButton.classList.add("btn-primary");
    document.querySelector('.trading-mode-card').classList.remove('trading-running');
    $parametersCard.show();
    resetAgentBtn.style.display = "none";
  }

  initializeTheme();
});
