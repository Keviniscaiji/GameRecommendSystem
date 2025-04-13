/**
 * 根据游戏数据生成卡片 HTML
 * @param {Object} rec 游戏数据对象
 * @returns {string} 返回 Bootstrap 卡片 HTML 字符串
 */
function generateGameCard(rec) {
  const description = rec.Description
    ? rec.Description.slice(0, 200) +
      (rec.Description.length > 200 ? "..." : "")
    : "No description available.";

  const rating = rec.Rating ? rec.Rating.toFixed(3) : "N/A";
  const weightedScore = rec.Weighted_Score
    ? rec.Weighted_Score.toFixed(3)
    : "N/A";
  const similarity = rec.Similarity ? rec.Similarity.toFixed(3) : "N/A";

  const cardHTML = `
      <div class="card game-card h-100">
        <img src="${
          rec.Image_url || "https://via.placeholder.com/350x150?text=No+Image"
        }" class="card-img-top" alt="${rec.Name}">
        <div class="card-body">
          <h5 class="card-title mb-1">${rec.Name}</h5>
          <p class="card-text mb-1">${description}</p>
          <small class="d-block mb-1">Release: ${
            rec.Release_Date || "N/A"
          }, Rating: ${rating}</small>
          <small class="d-block mb-1">Weighted Score: ${weightedScore}, Similarity: ${similarity}</small>
          <div>
            ${
              rec.Genres && rec.Genres.length > 0
                ? rec.Genres.map(
                    (tag) => `<span class="badge tag-badge">${tag}</span>`
                  ).join(" ")
                : ""
            }
          </div>
        </div>
      </div>
    `;

  // 判断是否存在合法链接
  const hasWebsite =
    typeof rec.Website === "string" && rec.Website.trim() !== "";

  return `
      <div class="col-md-4">
        ${
          hasWebsite
            ? `<a href="${rec.Website}" target="_blank" style="text-decoration: none; color: inherit;">${cardHTML}</a>`
            : cardHTML
        }
      </div>
    `;
}

async function searchUserProfile() {
  const steamId = document.getElementById("query").value.trim();
  const loader = document.getElementById("loaderContainer");
  const result = document.getElementById("userGameList");

  if (!steamId) {
    result.style.display = "block";
    result.innerHTML = `<div class="alert alert-warning" role="alert">
        Please enter a Steam ID!
      </div>`;
    loader.style.display = "none";
    return;
  }

  // 隐藏旧内容，显示 loading 动画
  result.style.display = "none";
  loader.style.display = "flex";

  try {
    const resp = await fetch(
      `/api/profile?steam_id=${encodeURIComponent(steamId)}`
    );
    const data = await resp.json();
    loader.style.display = "none";

    if (!data.games || data.games.length === 0) {
      result.innerHTML = `<div class="alert alert-info" role="alert">
          No games found in user's library.
        </div>`;
      result.style.display = "block";
      return;
    }

    // 按 playtime_hours 排序取用户最常玩的前 5 个游戏
    const top5 = [...data.games]
      .sort((a, b) => b.playtime_hours - a.playtime_hours)
      .slice(0, 5);

    const recArrays = await Promise.all(
      top5.map(async (g) => {
        const formData = new URLSearchParams();
        formData.append("game_id", g.app_id);
        formData.append("top_n", "20");
        if (
          document.getElementById("nicheCheckbox") &&
          document.getElementById("nicheCheckbox").checked
        ) {
          formData.append("niche_mode", "1");
        }
        const r = await fetch("/recommend", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: formData.toString(),
        });
        if (!r.ok) return [];
        const json = await r.json();
        return json.recommendations || [];
      })
    );

    // 合并推荐结果，去重（基于 rec.ID）
    const merged = new Map();
    recArrays.flat().forEach((rec) => {
      if (!merged.has(rec.ID)) merged.set(rec.ID, rec);
    });

    const final10 = Array.from(merged.values())
      .sort(() => Math.random() - 0.5)
      .slice(0, 10);

    // 使用 generateGameCard 生成推荐卡片 HTML
    let html = `
        <h2 class="mt-3">User Info</h2>
        <p><strong>Username:</strong> ${data.user_info.username}</p>
        <p><strong>Steam ID:</strong> ${data.user_info.steam_id}</p>
        <h3 class="mt-3">Based on your top 5 most‑played games, here are 10 picks for you:</h3>
        <div class="row g-3">
      `;
    final10.forEach((rec) => {
      html += generateGameCard(rec);
    });
    html += `</div>
        <button class="btn btn-secondary mt-3" onclick="document.getElementById('userGameList').style.display='none'">Close</button>`;
    result.innerHTML = html;
    result.style.display = "block";
  } catch (e) {
    console.error("Error:", e);
    loader.style.display = "none";
    result.innerHTML = `<div class="alert alert-danger" role="alert">
        There was an error loading the profile.
      </div>`;
    result.style.display = "block";
  }
}

let selectedGameId = null;

async function searchGames() {
  const query = document.getElementById("query").value;
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = "";
  const gameList = document.getElementById("gameList");
  gameList.innerHTML = "<p>Searching...</p>";
  gameList.style.display = "block";

  try {
    const response = await fetch("/search", {
      method: "POST",
      body: new URLSearchParams({ query: query }),
    });
    const data = await response.json();
    gameList.innerHTML = "";

    if (!data.games || data.games.length === 0) {
      gameList.innerHTML = "<p>No games found.</p>";
      document.getElementById("recommendSection").style.display = "none";
      return;
    }

    // 显示搜索结果，点击时选中游戏
    data.games.forEach((game) => {
      const div = document.createElement("div");
      div.className = "game-item";
      div.textContent = game.name;
      div.onclick = () => selectGame(game.appid, game.name);
      gameList.appendChild(div);
    });

    document.getElementById("recommendSection").style.display = "block";
  } catch (error) {
    console.error("Search failed:", error);
    gameList.innerHTML = "<p>Search failed. Please try again.</p>";
    document.getElementById("recommendSection").style.display = "none";
  }
}

function selectGame(appid, name) {
  selectedGameId = appid;
  document.getElementById("selectedGame").value = name;
  document.getElementById("game_id").value = appid;
  const gameList = document.getElementById("gameList");
  gameList.innerHTML = "";
  gameList.style.display = "none";
  document.getElementById("results").innerHTML = "";
}

// 监听推荐表单提交事件，调用 /recommend 接口并渲染推荐结果和 3D 图表
document.getElementById("recommendForm").addEventListener("submit", (e) => {
  e.preventDefault(); // 阻止默认提交
  if (!selectedGameId) {
    alert("Please select a game first.");
    return;
  }

  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = "<p>Loading recommendations and plot...</p>";
  resultsDiv.style.display = "block";

  const formData = new FormData(e.target);
  fetch("/recommend", {
    method: "POST",
    body: formData,
  })
    .then((response) => {
      if (!response.ok) {
        return response.json().then((errData) => {
          throw new Error(errData.error || response.statusText);
        });
      }
      return response.json();
    })
    .then((data) => {
      // 使用 generateGameCard 生成卡片列表
      let recHtml = `<h2 class="mt-3">Query Game: ${data.query_game}</h2>`;
      recHtml += `<div class="row g-3">`;
      data.recommendations.forEach((rec) => {
        recHtml += generateGameCard(rec);
      });
      recHtml += `</div>`;
      resultsDiv.innerHTML = recHtml;

      // 使用返回的 plot_data 绘制 3D 图表
      const plotData = data.plot_data;
      const queryPoints = plotData.filter((pt) => pt.Type === "Query");
      const recPoints = plotData.filter((pt) => pt.Type === "Recommendation");

      const traceQuery = {
        x: queryPoints.map((pt) => pt.PC1),
        y: queryPoints.map((pt) => pt.PC2),
        z: queryPoints.map((pt) => pt.PC3),
        mode: "markers+text",
        type: "scatter3d",
        name: "Query",
        marker: { size: 12, color: "#ff6b6b" },
        text: queryPoints.map((pt) => pt.Name),
        textposition: "top center",
        textfont: { color: "#ffffff" },
      };

      const traceRec = {
        x: recPoints.map((pt) => pt.PC1),
        y: recPoints.map((pt) => pt.PC2),
        z: recPoints.map((pt) => pt.PC3),
        mode: "markers+text",
        type: "scatter3d",
        name: "Recommendation",
        marker: { size: 8, color: "#4dabf7" },
        text: recPoints.map((pt) => pt.Name),
        textposition: "top center",
        textfont: { color: "#ffffff" },
      };

      const dataPlot = [traceQuery, traceRec];
      const layout = {
        title: {
          text: "Query & Recommendations (3D Visualization)",
          font: {
            family: "Arial, sans-serif",
            size: 18,
            color: "#ffffff",
          },
        },
        paper_bgcolor: "#1e2a38",
        plot_bgcolor: "#1e2a38",
        width: 1200,
        height: 800,
        scene: {
          bgcolor: "#1e2a38",
          xaxis: {
            title: { text: "PC1", font: { color: "#e0e0e0" } },
            color: "#e0e0e0",
            gridcolor: "#3c4c5a",
            zerolinecolor: "#5a6e83",
          },
          yaxis: {
            title: { text: "PC2", font: { color: "#e0e0e0" } },
            color: "#e0e0e0",
            gridcolor: "#3c4c5a",
            zerolinecolor: "#5a6e83",
          },
          zaxis: {
            title: { text: "PC3", font: { color: "#e0e0e0" } },
            color: "#e0e0e0",
            gridcolor: "#3c4c5a",
            zerolinecolor: "#5a6e83",
          },
        },
        legend: {
          font: { color: "#ffffff" },
          bgcolor: "rgba(30, 42, 56, 0.8)",
        },
      };

      const plotDiv = document.getElementById("plot");
      Plotly.newPlot(plotDiv, dataPlot, layout);
    })
    .catch((error) => {
      resultsDiv.innerHTML = `<p style="color: red;"><strong>Error:</strong> ${error.message}</p>`;
    });
});
