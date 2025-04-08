let selectedGameId = null;

async function searchGames() {
  const query = document.getElementById("query").value;
  const resultsDiv = document.getElementById("results"); // Clear previous results/plot
  resultsDiv.innerHTML = "";
  const gameList = document.getElementById("gameList");
  gameList.innerHTML = "<p>Searching...</p>"; // Indicate searching
  gameList.style.display = "block"; // Show the list area

  try {
    const response = await fetch("/search", {
      method: "POST",
      body: new URLSearchParams({ query: query }),
    });
    const data = await response.json();

    gameList.innerHTML = ""; // Clear searching message

    if (!data.games || data.games.length === 0) {
      gameList.innerHTML = "<p>No games found.</p>";
      document.getElementById("recommendSection").style.display = "none";
      // Keep gameList visible to show "No games found" message
      // gameList.style.display = "none"; // Optionally hide if no results found
      return;
    }

    // Display search results
    data.games.forEach((game) => {
      const div = document.createElement("div");
      div.className = "game-item";
      div.textContent = game.name;
      div.onclick = () => selectGame(game.appid, game.name);
      gameList.appendChild(div);
    });

    // Show recommendation section only after a successful search finds games
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
  gameList.innerHTML = ""; // Clear the list
  gameList.style.display = "none"; // Hide the list after selection
  document.getElementById("results").innerHTML = ""; // Clear previous results/plot
}

// --- Recommendation Form Submission ---
document.getElementById("recommendForm").addEventListener("submit", (e) => {
  e.preventDefault(); // Prevent default form submission
  if (!selectedGameId) {
    alert("Please select a game first.");
    return;
  }

  const resultsDiv = document.getElementById("results"); // Get results div reference
  resultsDiv.innerHTML = "<p>Loading recommendations and plot...</p>"; // Add loading indicator

  const formData = new FormData(e.target);

  fetch("/recommend", {
    // Use fetch API
    method: "POST",
    body: formData,
  })
    .then((response) => {
      if (!response.ok) {
        // Try to parse error message from backend JSON, otherwise use status text
        return response
          .json()
          .then((errData) => {
            throw new Error(
              errData.error ||
                response.statusText ||
                `HTTP error! Status: ${response.status}`
            );
          })
          .catch(() => {
            // If parsing error JSON fails, throw a generic error
            throw new Error(
              response.statusText || `HTTP error! Status: ${response.status}`
            );
          });
      }
      return response.json(); // Parse success JSON
    })
    .then((data) => {
      // --- Recommendations HTML ---
      let recHtml = `<h2>Query Game: ${data.query_game}</h2><ul>`;
      data.recommendations.forEach((rec) => {
        // Ensure description exists and safely slice it
        const description = rec.Description
          ? rec.Description.slice(0, 200) +
            (rec.Description.length > 200 ? "..." : "")
          : "No description available.";
        recHtml += `<li><strong>${rec.Name}</strong> (ID: ${
          rec.ID
        }, Weighted Score: ${rec.Weighted_Score.toFixed(
          3
        )}, Similarity: ${rec.Similarity.toFixed(3)})<br>
                      Release: ${
                        rec.Release_Date || "N/A"
                      }, Rating: ${rec.Rating.toFixed(3)}, Genres: ${
          rec.Genres && rec.Genres.length > 0 ? rec.Genres.join(", ") : "N/A"
        }<br>
                      Description: ${description}</li>`;
      });
      recHtml += `</ul>`;

      // Set recommendations HTML first, replacing the loading indicator
      resultsDiv.innerHTML = recHtml;

      // --- Plot Handling ---
      if (data.plot_html) {
        // Create a temporary container in memory for the plot HTML to parse it
        const plotContainer = document.createElement("div");
        plotContainer.innerHTML = data.plot_html; // Put Plotly's generated HTML here

        // Find the actual Plotly chart div (usually the first child element)
        const chartDiv = plotContainer.firstElementChild; // Use firstElementChild to skip text nodes

        // Find the script tag within the temporary container
        const scriptTag = plotContainer.querySelector("script");

        // Append the chart div to the results area if it exists
        if (chartDiv && chartDiv.nodeType === Node.ELEMENT_NODE) {
          // Optionally wrap the plot for spacing/styling
          const plotWrapper = document.createElement("div");
          plotWrapper.style.marginTop = "20px"; // Add some space above the plot
          plotWrapper.appendChild(chartDiv); // Move the chart div into the wrapper
          resultsDiv.appendChild(plotWrapper); // Add the wrapper (with chart) to the results area
        } else {
          console.warn("Plotly chart div not found in plot_html.");
        }

        // Find and execute the script tag if it exists
        if (scriptTag) {
          const newScript = document.createElement("script");
          // Copy type attribute if exists
          if (scriptTag.type) {
            newScript.type = scriptTag.type;
          }
          // Set the script content using textContent
          newScript.textContent = scriptTag.textContent;

          // Append the new script to the document body to execute it,
          // then immediately remove it to keep the DOM clean.
          document.body
            .appendChild(newScript)
            .parentNode.removeChild(newScript);
          console.log("Plotly script executed.");
        } else {
          console.warn("Plotly script tag not found in plot_html.");
        }
      } else {
        console.warn("plot_html not found in response data.");
        resultsDiv.innerHTML +=
          "<p>Plot could not be generated or was not provided.</p>"; // Inform user
      }
    })
    .catch((error) => {
      console.error("Error fetching/processing recommendations:", error);
      // Display the error in the results div
      resultsDiv.innerHTML = `<p style="color: red;"><strong>Error:</strong> ${error.message}</p>`;
    });
});
