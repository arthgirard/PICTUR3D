/* main.js */
document.addEventListener("DOMContentLoaded", function() {
  // Global variables
  let simulationFinishedUI = false;

  // Theme Toggle and Favicon Inversion Module
  const themeToggle = document.getElementById("themeToggle");
  const body = document.body;
  const navbar = document.getElementById("navbar");
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

  // Simulation Controls Module
  const startButton = document.getElementById("startSimulation");
  const modeSelect = document.getElementById("modeSelect");
  const simulationStatus = document.getElementById("simulationStatus");
  const startDateInput = document.getElementById("startDate");
  const endDateInput = document.getElementById("endDate");
  const stopLossInput = document.getElementById("stopLoss");
  const takeProfitInput = document.getElementById("takeProfit");
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

  // Extract the stop simulation UI logic into its own function.
  function stopSimulationUI() {
    simulationRunning = false;
    localStorage.setItem("simulationRunning", "false");
    startButton.disabled = true;
  
    if (resultsInterval) { clearInterval(resultsInterval); resultsInterval = null; }
    if (logsInterval) { clearInterval(logsInterval); logsInterval = null; }
    $("#liveActions").empty();
    if (window.btcPriceChart) { window.btcPriceChart.destroy(); window.btcPriceChart = null; }
    if (window.equityChart) { window.equityChart.destroy(); window.equityChart = null; }
    if (window.lossChart) { window.lossChart.destroy(); window.lossChart = null; }
    
    // Signal the backend to stop the simulation.
    fetch("/stop_simulation", { method: "POST" });
    
    // Adjust UI: show parameters card, hide results and logs.
    $parametersCard.slideDown(300);
    $resultsCard.slideUp(300);
    $logsCard.slideUp(300).promise().done(function() {
      startButton.textContent = "Start";
      startButton.classList.remove("btn-danger");
      startButton.classList.add("btn-primary");
      simulationStatus.textContent = "";
      resetAgentBtn.style.display = "none";
      waitForSimulationStop();
      startButton.disabled = false;
    });
  }

  function fetchResults() {
    fetch("/results")
      .then(response => response.json())
      .then(data => {
        updateCharts(data);
        if (firstLogReceived) {
          simulationStatus.innerHTML = `<strong>Balance:</strong> $${data.final_balance.toFixed(2)} | <strong>Trades:</strong> ${data.num_trades} | <strong>Return:</strong> ${data.percentage_return.toFixed(2)}%`;
        }
        // If simulation is finished and not yet finalized in UI:
        if (data.finished && !simulationFinishedUI) {
          simulationFinishedUI = true;
          stopSimulationUI();
          console.info("Simulation finalized automatically via final date.");
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
    // Check if the data object has the expected properties
    if (!data || !data.dates || data.dates.length === 0) {
      console.warn("No chart data available to update.");
      return;
    }
  
    // Equity Chart
    if (window.equityChart && window.equityChart.data) {
      window.equityChart.data.labels = data.dates;
      window.equityChart.data.datasets[0].data = data.asset_values;
      window.equityChart.update();
    } else {
      const ctx1 = document.getElementById("equityChart").getContext("2d");
      window.equityChart = new Chart(ctx1, {
        type: "line",
        data: {
          labels: data.dates,
          datasets: [{
            label: "Equity Curve (USD)",
            data: data.asset_values,
            borderColor: "rgba(75, 192, 192, 1)",
            fill: false
          }]
        },
        options: { responsive: true }
      });
    }
  
    // BTC Price Chart
    if (window.btcPriceChart && window.btcPriceChart.data) {
      window.btcPriceChart.data.labels = data.dates;
      window.btcPriceChart.data.datasets[0].data = data.btc_prices;
      if (data.trade_dates && data.trade_prices && data.trade_signals) {
        const tradeData = data.trade_dates.map((date, i) => ({
          x: date,
          y: data.trade_prices[i],
          signal: data.trade_signals[i]
        }));
        if (window.btcPriceChart.data.datasets.length < 2) {
          window.btcPriceChart.data.datasets.push({
            label: "Trades",
            data: tradeData,
            type: "scatter",
            parsing: { xAxisKey: "x", yAxisKey: "y" },
            pointRadius: 5,
            pointBackgroundColor: tradeData.map(pt =>
              pt.signal.toLowerCase().startsWith("buy") ? "green" : "red"
            ),
            showLine: false
          });
        } else {
          window.btcPriceChart.data.datasets[1].data = tradeData;
          window.btcPriceChart.data.datasets[1].pointBackgroundColor = tradeData.map(pt =>
            pt.signal.toLowerCase().startsWith("buy") ? "green" : "red"
          );
        }
      }
      window.btcPriceChart.update();
    } else {
      const ctx2 = document.getElementById("btcPriceChart").getContext("2d");
      let tradeData = [];
      if (data.trade_dates && data.trade_prices && data.trade_signals) {
        tradeData = data.trade_dates.map((date, i) => ({
          x: date,
          y: data.trade_prices[i],
          signal: data.trade_signals[i]
        }));
      }
      window.btcPriceChart = new Chart(ctx2, {
        type: "line",
        data: {
          labels: data.dates,
          datasets: [{
            label: "BTC Price (USD)",
            data: data.btc_prices,
            borderColor: "rgba(153, 102, 255, 1)",
            fill: false
          },
          {
            label: "Trades",
            data: tradeData,
            type: "scatter",
            parsing: { xAxisKey: "x", yAxisKey: "y" },
            pointRadius: 5,
            pointBackgroundColor: tradeData.map(pt =>
              pt.signal.toLowerCase().startsWith("buy") ? "green" : "red"
            ),
            showLine: false
          }]
        },
        options: { 
          responsive: true,
          scales: {
            x: {
              type: "time",
              time: {
                parser: "YYYY-MM-DD",
                tooltipFormat: "ll"
              }
            }
          }
        }
      });
    }
  
    // Loss Chart
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
            label: "Training Loss",
            data: data.losses,
            borderColor: "rgba(255, 99, 132, 1)",
            fill: false
          }]
        },
        options: { responsive: true }
      });
    }
  }
  

  function fetchAndUpdatePerformanceHistory() {
    fetch("/agent_performance")
      .then(response => response.json())
      .then(history => {
        const labels = history.map(item => item.timestamp);
        const netProfits = history.map(item => item.net_profit);
        if (window.performanceHistoryChart) {
          window.performanceHistoryChart.data.labels = labels;
          window.performanceHistoryChart.data.datasets[0].data = netProfits;
          window.performanceHistoryChart.update();
        } else {
          const ctx = document.getElementById("agentPerformanceChart").getContext("2d");
          window.performanceHistoryChart = new Chart(ctx, {
            type: "line",
            data: {
              labels: labels,
              datasets: [{
                label: "Bot Performance",
                data: netProfits,
                borderColor: "rgba(75, 192, 192, 1)",
                fill: false
              }]
            },
            options: {
              responsive: true,
              title: {
                display: true,
                text: "Historical Agent Performance"
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

  // Simulation start/stop handler
  startButton.addEventListener("click", function() {
  if (!simulationRunning) {
    // Reset the auto-stop flag for new simulation runs.
    simulationFinishedUI = false;
    
    $("#liveActions").empty();
    clearPreviousData();
    fetch("/clear_logs", { method: "POST" })
      .then(() => {
        simulationRunning = true;
        localStorage.setItem("simulationRunning", "true");
        startButton.textContent = "Stop";
        startButton.classList.remove("btn-primary");
        startButton.classList.add("btn-danger");
        startButton.style.display = "none";
        $parametersCard.slideUp(300);
        $resultsCard.hide();
        $logsCard.hide();
        simulationStatus.innerHTML = "Initializing agent <i class='fas fa-spinner fa-spin'></i>";
        const mode = modeSelect.value;
        const startDate = startDateInput.value;
        const endDate = endDateInput.value;
        const stopLoss = stopLossInput.value;
        const takeProfit = takeProfitInput.value;
        fetch("/start_simulation", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode: mode, start_date: startDate, end_date: endDate, stop_loss_pct: stopLoss, take_profit_pct: takeProfit })
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

  // Reset agent handler
  let resetInProgress = false;
  resetAgentBtn.addEventListener("click", function() {
    if (resetInProgress) return;
    resetInProgress = true;
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
        startButton.textContent = "Start";
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
        resetInProgress = false;
        resetAgentBtn.disabled = false;
        startButton.style.display = "inline-block";
      });
    } else {
      resetInProgress = false;
      resetAgentBtn.disabled = false;
      startButton.style.display = "inline-block";
    }
  });

  if (simulationRunning) {
    startButton.textContent = "Stop";
    startButton.classList.remove("btn-primary");
    startButton.classList.add("btn-danger");
    $parametersCard.hide();
    resetAgentBtn.style.display = "inline-block";
    fetch("/live_logs")
      .then(response => response.json())
      .then(data => {
        if (data && data.length > 0) {
          $resultsCard.show();
          $logsCard.show();
          lastKnownLogsCount = data.length;
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
    startButton.textContent = "Start";
    startButton.classList.remove("btn-danger");
    startButton.classList.add("btn-primary");
    $parametersCard.show();
    resetAgentBtn.style.display = "none";
  }

  initializeTheme();
});
