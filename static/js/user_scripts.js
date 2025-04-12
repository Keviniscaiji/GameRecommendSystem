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

  result.style.display = "none"; // Hide old content
  loader.style.display = "flex"; // Show loading spinner

  try {
    const resp = await fetch(`/api/profile?steam_id=${encodeURIComponent(steamId)}`);
    const data = await resp.json();
    loader.style.display = "none";

    if (!data.games || data.games.length === 0) {
      result.innerHTML = `<div class="alert alert-info" role="alert">
        No games found in user's library.
      </div>`;
      result.style.display = "block";
      return;
    }

    // Process top 5 most played games and render recommendations
    const top5 = [...data.games]
      .sort((a, b) => b.playtime_hours - a.playtime_hours)
      .slice(0, 5);

    const recArrays = await Promise.all(
      top5.map(async (g) => {
        const formData = new URLSearchParams();
        formData.append("game_id", g.app_id);
        formData.append("top_n", "20");
        if (document.getElementById("nicheCheckbox") && document.getElementById("nicheCheckbox").checked) {
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

    // Build HTML output using Bootstrap cards (based on steam.html style)
    let html = `
      <h2 class="mt-3">User Info</h2>
      <p><strong>Username:</strong> ${data.user_info.username}</p>
      <p><strong>Steam ID:</strong> ${data.user_info.steam_id}</p>
      <h3 class="mt-3">Based on your top 5 most‑played games, here are 10 picks for you:</h3>
      <div class="row g-3">
    `;

    final10.forEach((rec) => {
      const description = rec.Description
        ? rec.Description.slice(0, 200) + (rec.Description.length > 200 ? "..." : "")
        : "No description available.";
      html += `<div class="col-md-4">
                 <div class="card game-card">
                   <img src="${rec.Image_url || 'https://via.placeholder.com/350x150?text=No+Image'}" class="card-img-top" alt="${rec.Name}">
                   <div class="card-body">
                     <h5 class="card-title mb-1">${rec.Name}</h5>
                     <p class="card-text mb-1">${description}</p>
                     <small class="d-block mb-1">Release: ${rec.Release_Date || "N/A"}, Rating: ${rec.Rating ? rec.Rating.toFixed(3) : "N/A"}</small>
                     <small class="d-block mb-1">Weighted Score: ${rec.Weighted_Score.toFixed(3)}, Similarity: ${rec.Similarity.toFixed(3)}</small>
                     <div>
                       ${
                         rec.Genres && rec.Genres.length > 0
                           ? rec.Genres.map(tag => `<span class="badge tag-badge">${tag}</span>`).join(" ")
                           : ""
                       }
                     </div>
                   </div>
                 </div>
               </div>`;
    });
    html += `</div><button class="btn btn-secondary mt-3" onclick="document.getElementById('userGameList').style.display='none'">Close</button>`;
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

