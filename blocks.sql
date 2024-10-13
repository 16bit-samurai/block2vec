CREATE DATABASE defaults;

CREATE TABLE blocks(
    id INT AUTO_INCREMENT UNIQUE PRIMARY KEY,
    center INT NOT NULL,
    target INT NOT NULL 
);

SELECT count(*) FROM blocks;
