---
layout: archive
permalink: /posts/
title: Posts
author_profile: true
---

{% for post in site.posts %}
    {% include archive-single.html %}
{% endfor %}
