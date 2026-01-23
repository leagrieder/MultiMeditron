USER_ID=${USER_ID:0}
GROUP_ID=${GROUP_ID:0}
USERNAME=${USERNAME:root}

groupadd -f -g $GROUP_ID $USERNAME || true
id -u $USERNAME &>/dev/null || useradd -m -u $USER_ID -g $GROUP_ID $USERNAME || true

exec gosu $USER_ID:$GROUP_ID "$@"
