import requests # type: ignore
import json
import argparse
import os
from typing import Dict, List, Any, Optional, Tuple
import time

STEAM_API_KEY = None
class SteamProfileGenerator:
    def __init__(self, api_key: str):
        """
        Initialize the Steam Profile Generator with the provided API key.
        
        Args:
            api_key (str): Your Steam Web API key
        """
        self.api_key = api_key
        self.base_url = "https://api.steampowered.com"
        
    def get_user_profile(self, steam_id: str) -> Dict[str, Any]:
        """
        Get basic profile information for a user.
        
        Args:
            steam_id (str): 64-bit Steam ID
            
        Returns:
            Dict: User profile data
        """
        endpoint = f"{self.base_url}/ISteamUser/GetPlayerSummaries/v0002/"
        params = {
            "key": self.api_key,
            "steamids": steam_id
        }
        
        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            data = response.json()
            if data["response"]["players"]:
                return data["response"]["players"][0]
            else:
                print(f"No user data found for Steam ID: {steam_id}")
                return {}
        else:
            print(f"Error fetching user profile: {response.status_code}")
            return {}
    
    def get_owned_games(self, steam_id: str) -> Dict[str, Any]:
        """
        Get a list of games owned by the user.
        
        Args:
            steam_id (str): 64-bit Steam ID
            
        Returns:
            Dict: Owned games data
        """
        endpoint = f"{self.base_url}/IPlayerService/GetOwnedGames/v0001/"
        params = {
            "key": self.api_key,
            "steamid": steam_id,
            "include_appinfo": 1,
            "include_played_free_games": 1
        }
        
        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            print(f"Error fetching owned games: {response.status_code}")
            return {}
    
    def get_achievements_for_game(self, steam_id: str, app_id: int) -> Optional[Dict[str, Any]]:
        """
        Get achievements for a specific game.
        
        Args:
            steam_id (str): 64-bit Steam ID
            app_id (int): Application ID for the game
            
        Returns:
            Optional[Dict]: Achievement data if available
        """
        endpoint = f"{self.base_url}/ISteamUserStats/GetPlayerAchievements/v0001/"
        params = {
            "key": self.api_key,
            "steamid": steam_id,
            "appid": app_id,
            "l": "en"
        }
        
        try:
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                # Some games don't have achievements or profile is private
                return None
        except:
            # Sometimes the API returns an error for games with no achievements
            return None
    
    def get_game_schema(self, app_id: int) -> Optional[Dict[str, Any]]:
        """
        Get schema information about a game, including total achievements.
        
        Args:
            app_id (int): Application ID for the game
            
        Returns:
            Optional[Dict]: Game schema if available
        """
        endpoint = f"{self.base_url}/ISteamUserStats/GetSchemaForGame/v2/"
        params = {
            "key": self.api_key,
            "appid": app_id
        }
        
        try:
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except:
            return None
            
    def get_friend_list(self, steam_id: str) -> List[Dict[str, Any]]:
        """
        Get the user's friend list.
        
        Args:
            steam_id (str): 64-bit Steam ID
            
        Returns:
            List[Dict]: List of friends
        """
        endpoint = f"{self.base_url}/ISteamUser/GetFriendList/v0001/"
        params = {
            "key": self.api_key,
            "steamid": steam_id,
            "relationship": "friend"
        }
        
        try:
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                return response.json()["friendslist"]["friends"]
            else:
                print(f"Error fetching friend list: {response.status_code}")
                return []
        except:
            # Likely the profile is private
            print("Unable to fetch friend list. Profile may be private.")
            return []
    
    def calculate_achievement_percentage(self, steam_id: str, game: Dict[str, Any]) -> Optional[float]:
        """
        Calculate the percentage of achievements unlocked for a game.
        
        Args:
            steam_id (str): 64-bit Steam ID
            game (Dict): Game data
            
        Returns:
            Optional[float]: Percentage of achievements unlocked, or None if no achievements
        """
        app_id = game["appid"]
        achievement_data = self.get_achievements_for_game(steam_id, app_id)
        
        if not achievement_data or "achievements" not in achievement_data.get("playerstats", {}):
            return None
            
        achievements = achievement_data["playerstats"]["achievements"]
        if not achievements:
            return None
            
        total_achievements = len(achievements)
        unlocked_achievements = sum(1 for a in achievements if a.get("achieved", 0) == 1)
        
        if total_achievements > 0:
            return (unlocked_achievements / total_achievements) * 100
        return None
    
    def generate_user_profile(self, steam_id: str, max_games_for_achievements: int = 50) -> Dict[str, Any]:
        """
        Generate a complete user profile with all relevant information.
        
        Args:
            steam_id (str): 64-bit Steam ID
            max_games_for_achievements (int): Maximum number of games to process for achievements
            
        Returns:
            Dict: Complete user profile data
        """
        profile = {}
        
        # Get basic user info
        user_data = self.get_user_profile(steam_id)
        if not user_data:
            print(f"Could not retrieve profile for Steam ID: {steam_id}")
            return {}
            
        profile["user_info"] = {
            "steam_id": steam_id,
            "username": user_data.get("personaname", "Unknown"),
            "avatar": {
                "small": user_data.get("avatar"),
                "medium": user_data.get("avatarmedium"),
                "full": user_data.get("avatarfull")
            },
            "profile_url": user_data.get("profileurl"),
            "visibility": user_data.get("communityvisibilitystate", 0)
        }
        
        # Check if profile is public
        if profile["user_info"]["visibility"] != 3:
            print("Warning: This Steam profile is not public. Limited data will be available.")
        
        # Get owned games
        games_data = self.get_owned_games(steam_id)
        if not games_data:
            profile["games"] = []
            profile["total_games"] = 0
        else:
            profile["total_games"] = games_data.get("game_count", 0)
            games = games_data.get("games", [])
            
            # Process games
            processed_games = []
            for i, game in enumerate(sorted(games, key=lambda g: g.get("playtime_forever", 0), reverse=True)):
                game_info = {
                    "app_id": game["appid"],
                    "name": game.get("name", f"Unknown Game {game['appid']}"),
                    "playtime_minutes": game.get("playtime_forever", 0),
                    "playtime_hours": round(game.get("playtime_forever", 0) / 60, 2),
                    "icon_url": f"http://media.steampowered.com/steamcommunity/public/images/apps/{game['appid']}/{game.get('img_icon_url', '')}.jpg" if game.get('img_icon_url') else None,
                    "logo_url": f"http://media.steampowered.com/steamcommunity/public/images/apps/{game['appid']}/{game.get('img_logo_url', '')}.jpg" if game.get('img_logo_url') else None
                }
                
                # Get achievement percentage for top games only (to avoid too many API calls)
                if i < max_games_for_achievements:
                    achievement_percent = self.calculate_achievement_percentage(steam_id, game)
                    if achievement_percent is not None:
                        game_info["achievement_percentage"] = round(achievement_percent, 2)
                
                processed_games.append(game_info)
            
            profile["games"] = processed_games
        
        # Get friend list
        friends = self.get_friend_list(steam_id)
        profile["friends"] = [{"steam_id": friend["steamid"], "friend_since": friend["friend_since"]} for friend in friends]
        profile["total_friends"] = len(friends)
        
        return profile

    def save_profile_to_json(self, profile: Dict[str, Any], filename: str = None) -> str:
        """
        Save the user profile to a JSON file.
        
        Args:
            profile (Dict): User profile data
            filename (str, optional): Custom filename, default is steam_id.json
            
        Returns:
            str: Path to saved file
        """
        if not filename:
            steam_id = profile.get("user_info", {}).get("steam_id", "unknown")
            filename = f"{steam_id}_profile.json"
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2)
            
        print(f"Profile saved to {filename}")
        return filename


def main():
    parser = argparse.ArgumentParser(description='Generate a Steam user profile for recommendation systems')
    parser.add_argument('steam_id', help='64-bit Steam ID of the user')
    parser.add_argument('--api_key', help='Override the default Steam Web API key')
    parser.add_argument('--save', action='store_true', help='Save the profile to a JSON file')
    parser.add_argument('--output', help='Custom output file path (used with --save)')
    parser.add_argument('--max_games', type=int, default=50, help='Maximum number of games to process for achievements')
    
    args = parser.parse_args()
    
    # Get API key from arguments, environment variable, or the global constant
    api_key = args.api_key or os.environ.get('STEAM_API_KEY') or STEAM_API_KEY
    
    # Generate profile
    generator = SteamProfileGenerator(api_key)
    print(f"Generating profile for Steam ID: {args.steam_id}")
    profile = generator.generate_user_profile(args.steam_id, args.max_games)
    
    # Save to file if requested
    if args.save:
        generator.save_profile_to_json(profile, args.output)
    
    # Print summary
    print("\nProfile Summary:")
    user_info = profile.get("user_info", {})
    print(f"Username: {user_info.get('username', 'Unknown')}")
    print(f"Total Games: {profile.get('total_games', 0)}")
    print(f"Total Friends: {profile.get('total_friends', 0)}")
    
    # Print top 5 most played games
    print("\nTop 5 Most Played Games:")
    for i, game in enumerate(profile.get("games", [])[:5], 1):
        achievement_str = f", Achievements: {game.get('achievement_percentage', 'N/A')}%" if 'achievement_percentage' in game else ""
        print(f"{i}. {game.get('name', 'Unknown')} - {game.get('playtime_hours', 0)} hours{achievement_str}")
    
    return profile


if __name__ == "__main__":
    main()
