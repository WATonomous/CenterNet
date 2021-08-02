#!/bin/bash

if dpkg --compare-versions "$CODE_SERVER_VERSION" lt "3.11.0"; then
	export HASHED_PASSWORD=$(printf "$PASSWORD" | sha256sum | cut -d' ' -f1)
else
	export HASHED_PASSWORD=$(echo -n "$PASSWORD" | npx argon2-cli -e)
fi

/startup.sh
