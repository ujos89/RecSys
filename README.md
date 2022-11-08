# Recommendation System

## version

- pytorch 1.12.0
- pyg 2.1.0

## Data preprocessing

- Users

  - entry students include user information, 74298 users
  - filtering condition: member, loginCount > 3
  - columns: login_count, login_last, bookmark, follower, following, project, projectAll
- Items

  - projects created by filtered users, 457143 items
  - columns: category, comment_count, like_count, visit_count, user_id, create_date, update_date
- Session

  - user likes item (project), 2137131 sessions
  - columns: user_id, item_id, time
