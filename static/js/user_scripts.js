async function searchUserProfile() {
  const steamId = document.getElementById("query").value.trim();
  const loader = document.getElementById("loaderContainer");
  const result = document.getElementById("userGameList");

  if (!steamId) {
    result.style.display = "block";
    result.innerHTML = "<p>Please enter a Steam ID!</p>";
    loader.style.display = "none";
    return;
  }

  result.style.display = "none"; // 隐藏旧内容
  loader.style.display = "flex"; // 显示加载动画

  try {
    const resp = await fetch(
      `/api/profile?steam_id=${encodeURIComponent(steamId)}`
    );
    const data = await resp.json();
    loader.style.display = "none";

    if (!data.games || data.games.length === 0) {
      result.innerHTML = "<p>No games found in user's library.</p>";
      result.style.display = "block";
      return;
    }

    // 处理 Top 5 最多游戏推荐逻辑，并渲染内容
    const top5 = [...data.games]
      .sort((a, b) => b.playtime_hours - a.playtime_hours)
      .slice(0, 5);

    const recArrays = await Promise.all(
      top5.map(async (g) => {
        const formData = new URLSearchParams();
        formData.append("game_id", g.app_id);
        formData.append("top_n", "20");
        if (document.getElementById("nicheCheckbox").checked) {
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

    const merged = new Map();
    recArrays.flat().forEach((rec) => {
      if (!merged.has(rec.ID)) merged.set(rec.ID, rec);
    });

    const final10 = Array.from(merged.values())
      .sort(() => Math.random() - 0.5)
      .slice(0, 10);

    let html = `
      <h2>User Info</h2>
      <p>Username: ${data.user_info.username}</p>
      <p>Steam&nbsp;ID: ${data.user_info.steam_id}</p>
      <h3>Based on your top&nbsp;5 most‑played games, here are 10 picks for you:</h3>
      <ul>
    `;

    final10.forEach((rec) => {
      const description = rec.Description
        ? rec.Description.slice(0, 200) +
          (rec.Description.length > 200 ? "..." : "")
        : "No description available.";

      html += `<li>
        <strong>${rec.Name}</strong> (ID: ${
        rec.ID
      }, Weighted Score: ${rec.Weighted_Score.toFixed(3)}, 
        Review Number: ${
          rec.Num_of_reviews
        }, Similarity: ${rec.Similarity.toFixed(3)})<br>
        Release: ${rec.Release_Date || "N/A"}, 
        Rating: ${rec.Rating ? rec.Rating.toFixed(3) : "N/A"}, 
        Genres: ${
          rec.Genres && rec.Genres.length > 0 ? rec.Genres.join(", ") : "N/A"
        }<br>
        <em>${description}</em>
      </li>`;
    });

    html += `</ul><button onclick="document.getElementById('userGameList').style.display='none'">Close</button>`;
    result.innerHTML = html;
    result.style.display = "block";
  } catch (e) {
    console.error("Error:", e);
    loader.style.display = "none";
    result.innerHTML = "<p>There was an error loading the profile.</p>";
    result.style.display = "block";
  }
}
